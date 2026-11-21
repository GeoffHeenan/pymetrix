"""Co-skewness and co-kurtosis (Dataset.coskewness, Dataset.cokurtosis) and systematic co-moments with a market series
(Dataset.systematic_comoments), and their place in the Correlations dialog.

Checked: every element against a direct numpy computation by hand (loops over i, j, k, l of means of products of
deviations, 1e-12), in both forms (the matrices M3 and M4, and the pairs), standardized or not; M3 and M4 are unchanged by
any permutation of their indices (as tensors); the diagonal is each series' skewness and (not excess) kurtosis with
the T denominator, scipy's skew and kurtosis with bias=True, and converts exactly to describe()'s bias-corrected
skewness and excess kurtosis; the standardized versions are unchanged by a change of location and of (positive) scale
of any series, and a negative scale flips the sign of terms with that series an odd number of times; observations
are the periods every series has within the active sample; the systematic co-moments against the definitions by hand
(and beta against a least-squares slope); errors for bad input; the dialog records code that gives the same table,
and round-trips through a project."""

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pymetrix import Dataset
from pymetrix.errors import DataError


def _data(n=240, seed=5):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(5, size=(n, 3))
    frame = pd.DataFrame({"a": z[:, 0] + 0.3 * z[:, 0] ** 2, "b": 0.5 * z[:, 0] + z[:, 1], "c": np.exp(0.4 * z[:, 2]) + 0.2 * z[:, 1]},
                         index=pd.period_range("1990-01", periods=n, freq="M", name="time"))
    frame.iloc[3, 1] = np.nan  # one gap: co-moments use the periods every series has
    return Dataset.from_frame(frame)


def _by_hand(x: np.ndarray, order: int, standardized: bool) -> np.ndarray:
    """The co-moment tensor, element by element."""
    n, k = x.shape
    d = x - x.mean(axis=0)
    if standardized:
        d = d / np.sqrt((d ** 2).sum(axis=0) / n)
    out = np.empty((k,) * order)
    for idx in itertools.product(range(k), repeat=order):
        out[idx] = np.mean(np.prod([d[:, i] for i in idx], axis=0))
    return out


def _values(table) -> np.ndarray:
    return table.to_frame().to_numpy(float)


@pytest.mark.parametrize("standardized", [True, False])
def test_against_a_direct_computation(standardized):
    data = _data()
    x = data.frame[["a", "b", "c"]].dropna().to_numpy()
    for order, method in ((3, data.coskewness), (4, data.cokurtosis)):
        tensor = _by_hand(x, order, standardized)
        matrix = method(standardized=standardized)
        np.testing.assert_allclose(_values(matrix), tensor.reshape(3, -1), rtol=1e-12, atol=1e-12)
        frame = matrix.to_frame()
        assert list(frame.index) == ["a", "b", "c"]
        assert list(frame.columns) == [",".join(c) for c in itertools.product("abc", repeat=order - 1)]
        pairs = method(standardized=standardized, form="pairs").to_frame()
        expected = np.array([[tensor[(i,) * (order - 1) + (j,)] for j in range(3)] for i in range(3)])
        np.testing.assert_allclose(pairs.to_numpy(float), expected, rtol=1e-12, atol=1e-12)
        assert list(pairs.index) == list(pairs.columns) == ["a", "b", "c"]


def test_symmetry_and_the_univariate_diagonal():
    data = _data()
    x = data.frame[["a", "b", "c"]].dropna()
    T = len(x)
    m3 = _values(data.coskewness()).reshape(3, 3, 3)
    m4 = _values(data.cokurtosis()).reshape(3, 3, 3, 3)
    for perm in itertools.permutations(range(3)):
        np.testing.assert_allclose(m3, m3.transpose(perm), rtol=0, atol=1e-13)
    for perm in itertools.permutations(range(4)):
        np.testing.assert_allclose(m4, m4.transpose(perm), rtol=0, atol=1e-13)
    g1 = np.array([m3[i, i, i] for i in range(3)])
    b2 = np.array([m4[i, i, i, i] for i in range(3)])
    np.testing.assert_allclose(g1, stats.skew(x.to_numpy(), bias=True), rtol=1e-12)
    np.testing.assert_allclose(b2, stats.kurtosis(x.to_numpy(), fisher=False, bias=True), rtol=1e-12)
    # describe(): bias-corrected skewness and excess kurtosis, from these by the standard formulas
    described = data.describe(["a", "b", "c"]).to_frame()
    complete = Dataset.from_frame(x).describe(["a", "b", "c"]).to_frame()  # describe uses each series' own periods
    G1 = g1 * np.sqrt(T * (T - 1)) / (T - 2)
    G2 = ((T + 1) * (b2 - 3) + 6) * (T - 1) / ((T - 2) * (T - 3))
    np.testing.assert_allclose(complete["skewness"].to_numpy(float), G1, rtol=1e-12)
    np.testing.assert_allclose(complete["kurtosis"].to_numpy(float), G2, rtol=1e-12)
    assert described.loc["a", "count"] == T + 1  # the gap in b is in a's own periods
    pairs = data.coskewness(form="pairs").to_frame().to_numpy(float)
    np.testing.assert_allclose(np.diag(pairs), g1, rtol=1e-12)


def test_standardized_versions_do_not_depend_on_location_or_scale():
    data = _data()
    moved = Dataset.from_frame(data.frame.assign(a=3.0 * data.frame["a"] - 7.0, b=0.01 * data.frame["b"] + 100.0, c=data.frame["c"]))
    for method in ("coskewness", "cokurtosis"):
        np.testing.assert_allclose(_values(getattr(moved, method)()), _values(getattr(data, method)()), rtol=1e-9, atol=1e-12)
        unstd_old, unstd_new = _values(getattr(data, method)(standardized=False)), _values(getattr(moved, method)(standardized=False))
        assert not np.allclose(unstd_old, unstd_new)
    flipped = Dataset.from_frame(data.frame.assign(a=-data.frame["a"]))
    m3, f3 = _values(data.coskewness()).reshape(3, 3, 3), _values(flipped.coskewness()).reshape(3, 3, 3)
    for idx in itertools.product(range(3), repeat=3):
        sign = (-1) ** idx.count(0)
        assert f3[idx] == pytest.approx(sign * m3[idx], abs=1e-12)


def test_the_active_sample_and_the_series_named():
    data = _data()
    within = data.with_sample(("1995-01", "2005-12"))
    rows = within.in_sample()[["a", "c"]].dropna().to_numpy()
    assert len(rows) == 132
    np.testing.assert_allclose(_values(within.coskewness(["a", "c"])), _by_hand(rows, 3, True).reshape(2, -1), rtol=1e-12, atol=1e-12)
    assert "a,c" in within.coskewness(["a", "c"]).to_frame().columns
    one = data.cokurtosis("c", form="pairs").to_frame()
    assert one.shape == (1, 1)


def test_systematic_comoments_against_the_definitions():
    data = _data()
    x = data.frame[["a", "c", "b"]].dropna()
    d = x - x.mean()
    m = d["b"].to_numpy()
    table = data.systematic_comoments(["a", "c"], market="b").to_frame()
    assert list(table.index) == ["a", "c"]
    assert list(table.columns) == ["beta", "coskewness", "cokurtosis", "coskewness beta", "cokurtosis beta"]
    for name in ("a", "c"):
        s = d[name].to_numpy()
        sd_s, sd_m = np.sqrt(np.mean(s ** 2)), np.sqrt(np.mean(m ** 2))
        expected = [np.mean(s * m) / np.mean(m ** 2), np.mean(s * m ** 2) / (sd_s * sd_m ** 2), np.mean(s * m ** 3) / (sd_s * sd_m ** 3),
                    np.mean(s * m ** 2) / np.mean(m ** 3), np.mean(s * m ** 3) / np.mean(m ** 4)]
        np.testing.assert_allclose(table.loc[name].to_numpy(float), expected, rtol=1e-12)
        slope = np.polyfit(x["b"].to_numpy(), x[name].to_numpy(), 1)[0]
        assert table.loc[name, "beta"] == pytest.approx(slope, rel=1e-10)
    # the market's own co-skewness with itself is its skewness: the pairs table agrees
    pairs = data.coskewness(["a", "b"], form="pairs").to_frame()
    two = Dataset.from_frame(x[["a", "b"]]).systematic_comoments("a", market="b").to_frame()
    assert two.loc["a", "coskewness"] == pytest.approx(pairs.loc["b", "a"], rel=1e-12)


def test_errors():
    data = _data()
    with pytest.raises(DataError, match="form is"):
        data.coskewness(form="tensor")
    with pytest.raises(DataError, match="no series named"):
        data.cokurtosis(["a", "zz"])
    with pytest.raises(DataError, match="does not vary"):
        Dataset.from_frame(data.frame.assign(k=1.0)).coskewness(["a", "k"])
    with pytest.raises(DataError, match="market series"):
        data.systematic_comoments(["a", "b"], market="b")
    with pytest.raises(DataError, match="too few"):
        Dataset.from_frame(data.frame.iloc[:2]).coskewness()


def test_the_correlations_dialog(tmp_path):
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import Correlations
    from pymetrix.testing import check_project

    data = _data()
    s = Session()
    s.objects["data"] = data
    cases = [
        (Correlations(series="a, b", show="coskewness", name="m3"), "m3 = data.coskewness(('a', 'b'))", data.coskewness(["a", "b"])),
        (Correlations(show="cokurtosis", form="pairs", standardized=False, name="k"),
         "k = data.cokurtosis(standardized=False, form='pairs')", data.cokurtosis(standardized=False, form="pairs")),
        (Correlations(series="a, c", show="systematic", market="b", name="sys"),
         "sys = data.systematic_comoments(('a', 'c'), 'b')", data.systematic_comoments(["a", "c"], market="b")),
    ]
    for command, code, expected in cases:
        assert command.validate(s) == []
        result = s.run(command)
        assert code in s.script()
        np.testing.assert_array_equal(_values(result), _values(expected))
    namespace = {"data": data}
    exec(s.script(), namespace)  # the recorded code reproduces the tables
    np.testing.assert_array_equal(_values(namespace["sys"]), _values(s["sys"]))
    assert Correlations(series="a", show="systematic").validate(s) == ["choose the series and the market series to relate them to"]
    assert Correlations(series="a, b", show="systematic", market="b").validate(s) == \
        ["the market series cannot also be one of the series related to it"]
    check_project(s, Correlations(series="a, b, c", show="cokurtosis", name="m4"), tmp_path)
