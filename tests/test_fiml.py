"""Full-information maximum likelihood for simultaneous systems, checked independently: exactly identified equations
give 2SLS; one over-identified structural equation beside its unrestricted reduced form gives LIML (linearmodels'
IVLIML); with no endogenous regressors FIML is iterated SUR, standard errors included; an identity enters the
Jacobian (the log likelihood by hand, above 3SLS's); refusals."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import FIML, SUR, Dataset, Equation, System, ThreeSLS, TwoSLS
from pymetrix.errors import SpecificationError


def _market(n=600, seed=0, extra=False):
    """Demand q = 10 - 1.0 p + 0.5 income + u1; supply q = 2 + 0.8 p + 0.6 rain (+ 0.4 wind) + u2."""
    rng = np.random.default_rng(seed)
    income, rain, wind = rng.normal(5, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
    u1, u2 = rng.multivariate_normal([0, 0], [[1.0, 0.4], [0.4, 0.8]], n).T
    sup = 2 + 0.6 * rain + (0.4 * wind if extra else 0) + u2
    dem = 10 + 0.5 * income + u1
    p = (dem - sup) / (1.0 + 0.8)
    q = dem - 1.0 * p
    frame = pd.DataFrame({"id": range(n), "q": q, "p": p, "income": income, "rain": rain, "wind": wind})
    return Dataset.from_frame(frame, entity="id")


def test_exactly_identified_equations_give_2sls():
    data = _market()
    sys = System({"demand": Equation("q ~ p + income"), "supply": Equation("q ~ p + rain")}, endogenous=["p"])
    fiml = sys.fit(data, method=FIML())
    tsls = sys.fit(data, method=TwoSLS())
    assert fiml.params.values == pytest.approx(tsls.params.values, abs=1e-6)
    assert fiml.params.as_dict()["demand.p"] == pytest.approx(-1.0, abs=0.15)


def test_one_structural_equation_and_the_reduced_form_give_liml():
    pytest.importorskip("linearmodels")
    from linearmodels.iv import IVLIML

    data = _market(extra=True)
    f = data.frame.reset_index()
    ref = IVLIML(f["q"], pd.DataFrame({"const": 1.0, "income": f["income"]}), f[["p"]], f[["rain", "wind"]]).fit(cov_type="unadjusted")
    sys = System({"demand": Equation("q ~ p + income"), "price": Equation("p ~ income + rain + wind")}, endogenous=["q"])
    fiml = sys.fit(data, method=FIML())
    ours = fiml["demand"].params.as_dict()
    assert [ours["const"], ours["p"], ours["income"]] == pytest.approx(ref.params.reindex(["const", "p", "income"]).to_numpy(), rel=1e-6)
    se = dict(zip(fiml["demand"].params.names, fiml["demand"].params.se))
    assert se["p"] == pytest.approx(ref.std_errors["p"], rel=0.05)  # the same limit; LIML's k-class formula differs in finite samples


def test_without_endogenous_regressors_fiml_is_iterated_sur():
    rng = np.random.default_rng(4)
    n = 300
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    e = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], n)
    data = Dataset.from_frame(pd.DataFrame({"id": range(n), "y1": 1 + 2 * x1 + e[:, 0], "y2": -1 + 0.5 * x2 + e[:, 1],
                                            "x1": x1, "x2": x2}), entity="id")
    sys = System({"a": Equation("y1 ~ x1"), "b": Equation("y2 ~ x2")})
    fiml = sys.fit(data, method=FIML())
    sur = sys.fit(data, method=SUR(iterate=True, tol=1e-12))
    assert fiml.params.values == pytest.approx(sur.params.values, abs=1e-6)
    assert fiml.params.se == pytest.approx(sur.params.se, rel=1e-2)  # the observed concentrated Hessian against GLS's expected information
    assert fiml.stats["loglike"] == pytest.approx(sur.stats["loglike"], rel=1e-9) and fiml.stats["log_jacobian"] == pytest.approx(0, abs=1e-12)


def test_an_identity_enters_the_jacobian():
    rng = np.random.default_rng(5)
    n = 400
    g, r = rng.normal(10, 1, n), rng.normal(0, 1, n)
    u1, u2 = rng.normal(size=n), rng.normal(size=n) * 0.5
    # c = 1 + 0.6 y + u1; i = 2 - 0.5 r + 0.1 y + u2; y = c + i + g  =>  y = (3 - 0.5 r + g + u1 + u2) / 0.3
    y = (3 - 0.5 * r + g + u1 + u2) / 0.3
    c = 1 + 0.6 * y + u1
    i = y - c - g
    data = Dataset.from_frame(pd.DataFrame({"id": range(n), "c": c, "i": i, "y": y, "g": g, "r": r}), entity="id")
    sys = System([Equation("c ~ y"), Equation("i ~ r + y"), Equation("y = c + i + g")])
    fit = sys.fit(data, method=FIML())
    est = fit.params.as_dict()
    a, b = est["c.y"], est["i.y"]
    U = fit.resid.to_numpy()
    J = np.array([[1, 0, -a], [0, 1, -b], [-1, -1, 1]])  # d(residual) / d(c, i, y), the identity's row last
    ll = n * np.log(abs(np.linalg.det(J))) - n / 2 * (2 * (1 + np.log(2 * np.pi)) + np.linalg.slogdet(U.T @ U / n)[1])
    assert fit.stats["loglike"] == pytest.approx(ll, rel=1e-10)
    assert a == pytest.approx(0.6, abs=0.05) and est["i.r"] == pytest.approx(-0.5, abs=0.1)
    three = sys.fit(data, method=ThreeSLS())  # the consumption equation is over-identified: FIML and 3SLS differ, FIML's likelihood higher
    t = three.params.as_dict()
    U3 = three.resid.to_numpy()
    J3 = np.array([[1, 0, -t["c.y"]], [0, 1, -t["i.y"]], [-1, -1, 1]])
    ll3 = n * np.log(abs(np.linalg.det(J3))) - n / 2 * (2 * (1 + np.log(2 * np.pi)) + np.linalg.slogdet(U3.T @ U3 / n)[1])
    assert fit.stats["loglike"] >= ll3 and fit.params.values == pytest.approx(three.params.values, abs=0.05)


def test_refusals():
    data = _market()
    with pytest.raises(SpecificationError, match="as many equations"):
        System({"demand": Equation("q ~ p + income")}, endogenous=["p"]).fit(data, method=FIML())
    with pytest.raises(SpecificationError, match="classic"):
        System({"demand": Equation("q ~ p + income"), "supply": Equation("q ~ p + rain")}, endogenous=["p"]).fit(data, method=FIML(cov="white"))
    with pytest.raises(SpecificationError, match="restrictions"):
        System({"demand": Equation("q ~ p + income"), "supply": Equation("q ~ p + rain")}, endogenous=["p"],
               restrictions="demand.income = 0.5").fit(data, method=FIML())
