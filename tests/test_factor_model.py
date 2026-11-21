"""Maximum-likelihood factor analysis, against R's factanal (stats, R 4.3.3) on pymetrix's mroz data: uniquenesses,
loadings unrotated and rotated (varimax, promax), the test of the number of factors, and regression and Bartlett
scores; the rotations' defining properties."""

import numpy as np
import pytest

from pymetrix import Dataset, FactorModel
from pymetrix.factors import promax, varimax

VARIABLES = ["educ", "motheduc", "fatheduc", "huseduc", "age", "exper", "kidsge6"]

# factanal(x, factors=k, rotation=, scores=) on mroz[, VARIABLES]; loadings column by column, the first two rows' scores
R = {
    (2, "none"): {
        "uniq": [0.3351353661, 0.6183998530, 0.6187157416, 0.5277499268, 0.5161360036, 0.7306622228, 0.6742743407],
        "load": [0.7932958582, 0.6114318394, 0.6173241483, 0.6839489606, -0.2913101759, -0.0669133501, 0.0334850170,
                 0.1885396464, -0.0880704020, -0.0138754941, 0.0668014113, 0.6316664154, 0.5146467172, -0.5697411014],
        "reg": [0.0855135706, -0.3872314142, -0.2308070625, -1.0628628569],
        "bart": [0.1060852065, -0.4803860280, -0.3631245420, -1.6721827483]},
    (2, "varimax"): {
        "load": [0.8103325321, 0.5961902326, 0.6110586280, 0.6869970829, -0.2123585682, -0.0038520436, -0.0360261125,
                 0.0907011818, -0.1617482503, -0.0888199672, -0.0168409442, 0.6623955898, 0.5189641623, -0.5695860674],
        "reg": [0.0568203671, -0.5135703359, -0.2394909534, -1.0079043262],
        "bart": [0.0611537392, -0.6801083413, -0.3733279045, -1.6013801934]},
    (2, "promax"): {
        "load": [0.8312420048, 0.5658813284, 0.5953851559, 0.6859780054, -0.0806151393, 0.0999200755, -0.1500582217,
                 0.1064024764, -0.1554549526, -0.0805983620, -0.0056868927, 0.6740736163, 0.5308439170, -0.5833008945],
        "reg": [0.0091199282, -0.7168924659, -0.2440484814, -1.0397302987],  # without the factor correlations (R 4.3)
        "bart": [0.0672755296, -0.6537220281, -0.3780801468, -1.4375584700]},
    (3, "varimax"): {
        "uniq": [0.0050000000, 0.5440386903, 0.2250408847, 0.5994215542, 0.5077393635, 0.7410456197, 0.6535287713],
        "load": [0.9626366083, 0.3083572011, 0.2395269133, 0.5706313566, -0.1368470812, 0.0565320180, -0.0058824444,
                 0.2526313770, 0.5847048074, 0.8465871924, 0.2655572386, -0.1340418886, -0.0857125302, -0.0553175291,
                 0.0671438449, -0.1378321268, -0.0295986436, -0.0666075954, 0.6749562615, 0.4984079865, -0.5859829369]},
}


@pytest.fixture(scope="module")
def mroz():
    return Dataset.example("mroz")


@pytest.mark.parametrize("k, rotation", list(R))
def test_factanal(mroz, k, rotation):
    fit = FactorModel(VARIABLES, k, rotation=rotation).fit(mroz)
    ref = R[(k, rotation)]
    tol = 1e-8 if k == 2 else 2e-4  # k = 3 is a Heywood case: the optimisers stop at slightly different points
    if "uniq" in ref:
        assert fit.uniquenesses.to_numpy() == pytest.approx(ref["uniq"], abs=tol)
    assert fit.loadings.to_numpy().T.ravel() == pytest.approx(ref["load"], abs=tol)
    if "bart" in ref:
        assert fit.scores("bartlett").iloc[:2].to_numpy().T.ravel() == pytest.approx(ref["bart"], abs=1e-7)
        regression = fit.scores("regression").iloc[:2].to_numpy()
        if rotation == "promax":  # Thomson's scores for correlated factors: R 4.3's times the factors' correlations
            np.testing.assert_allclose(regression, np.reshape(ref["reg"], (2, 2), order="F") @ fit.factor_correlation.to_numpy(), atol=1e-7)
        else:
            assert regression.T.ravel() == pytest.approx(ref["reg"], abs=1e-7)
    stat = {2: (133.11240387, 8), 3: (4.20226996, 3)}[k]
    assert fit.test.statistic == pytest.approx(stat[0], abs=1e-4 if k == 2 else 2e-3) and fit.test.df == stat[1]
    assert (fit.stats["heywood"] == ["educ"]) == (k == 3)


def test_rotations_keep_the_common_part(mroz):
    L0 = FactorModel(VARIABLES, 3).fit(mroz).loadings.to_numpy()
    common = L0 @ L0.T
    for rotation in ("varimax", "promax"):
        fit = FactorModel(VARIABLES, 3, rotation=rotation).fit(mroz)
        L = fit.loadings.to_numpy()
        phi = np.eye(3) if fit.factor_correlation is None else fit.factor_correlation.to_numpy()
        np.testing.assert_allclose(L @ phi @ L.T, common, atol=1e-8)  # the same communalities and fitted correlations
        assert np.all(np.sum(L, axis=0) > 0) and np.all(np.diff(np.sum(L**2, axis=0)) <= 1e-12)  # signed, ordered
    rng = np.random.default_rng(0)
    L = rng.normal(size=(8, 3))
    R_, T = varimax(L)
    np.testing.assert_allclose(T.T @ T, np.eye(3), atol=1e-12)  # orthogonal
    P, U = promax(L)
    np.testing.assert_allclose(P, L @ U, atol=1e-12)
    assert np.allclose(np.diag(np.linalg.inv(U.T @ U)), 1)  # unit-variance factors


def test_refusals_and_views(mroz):
    with pytest.raises(ValueError, match="too many"):
        FactorModel(VARIABLES[:4], 3).fit(mroz)
    with pytest.raises(ValueError, match="DynamicFactor"):
        FactorModel(VARIABLES, 2, dynamics=1).fit(mroz)
    fit = FactorModel(VARIABLES, 2, rotation="promax").fit(mroz)
    text = str(fit.summary())
    assert "Factor correlations" in text and "2 factors are sufficient" in text
    assert fit.communalities.to_numpy() == pytest.approx(1 - fit.uniquenesses.to_numpy())
    orthogonal = FactorModel(VARIABLES, 2, rotation="varimax").fit(mroz)  # with uncorrelated factors the shares add up
    assert orthogonal.explained["proportion"].sum() == pytest.approx(np.mean(orthogonal.communalities), rel=1e-5)  # at the optimum, to its tolerance
