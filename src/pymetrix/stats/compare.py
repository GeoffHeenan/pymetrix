"""Comparisons across fitted objects."""

from __future__ import annotations

from typing import Literal

import numpy as np

from pymetrix._lazy import stats as sps

from .._api import api
from ..errors import SpecificationError
from ..estimation import FittedEquation
from ..output import Table
from .base import TestResult


@api("pymetrix:compare")
def compare(*fits: FittedEquation, test: Literal["lr", "hausman"] | None = None, sigma: str = "efficient") -> Table | TestResult:
    """Without ``test``: a table of fit statistics. ``test="lr"``: nested likelihood ratio.
    ``test="hausman"``: consistent fit first, efficient fit second (see ``_hausman`` for ``sigma``)."""
    if test is None:
        keys = ["nobs", "k", "r2", "aic", "bic", "loglike"]
        rows = [[f.stats.get(k) for k in keys] for f in fits]
        return Table(keys, rows, index=[f.name for f in fits], title="Model comparison")
    if test == "lr":
        if len(fits) != 2:
            raise SpecificationError("the likelihood-ratio test compares two nested fits")
        small, big = sorted(fits, key=lambda f: f.stats["k"])
        stat = 2 * (big.stats["loglike"] - small.stats["loglike"])
        df = int(big.stats["k"] - small.stats["k"])
        return TestResult("Likelihood ratio", "the restricted model is adequate", stat, df, float(sps.chi2.sf(stat, df)), "chi2")
    if test == "hausman":
        return _hausman(*fits, sigma=sigma)
    raise SpecificationError(f"unknown comparison {test!r}")


def _hausman(*fits: FittedEquation, sigma: str = "efficient") -> TestResult:
    """Hausman test: ``compare(consistent, efficient, test="hausman")``, e.g. fixed effects
    against random effects, on the coefficients the fits share (not the constant).

    ``sigma="efficient"`` (default; Stata's ``sigmamore``) rescales the consistent fit's
    covariance to the efficient fit's error variance, so the covariance difference is
    positive semi-definite in finite samples. ``"separate"`` uses each fit's own covariance.
    """
    if len(fits) != 2:
        raise SpecificationError("the Hausman test compares two fits: the consistent one first, then the efficient one")
    if sigma not in ("efficient", "separate"):
        raise SpecificationError("sigma must be 'efficient' or 'separate'")
    a, b = fits
    if a is b:
        raise SpecificationError("the Hausman test compares two different fits: the consistent one, then the efficient one")
    common = [n for n in a.params.names if n in b.params.names and n != "const"]
    if not common:
        raise SpecificationError("the two fits share no coefficients")
    ia = [a.params.names.index(n) for n in common]
    ib = [b.params.names.index(n) for n in common]
    diff = a.params.values[ia] - b.params.values[ib]
    Va = a.params.cov[np.ix_(ia, ia)]  # type: ignore[index]
    if sigma == "efficient":
        if "se_regression" not in a.stats or "se_regression" not in b.stats:
            raise SpecificationError("sigma='efficient' needs both fits to report se_regression; use sigma='separate'")
        Va = Va * (b.stats["se_regression"] / a.stats["se_regression"]) ** 2
    V = Va - b.params.cov[np.ix_(ib, ib)]  # type: ignore[index]
    stat = float(diff @ np.linalg.pinv(V) @ diff)
    df = int(np.linalg.matrix_rank(V))
    if df == 0:
        raise SpecificationError("the two fits have the same covariance on their shared coefficients, so there is nothing to "
                                 "test: compare a consistent fit with a different, efficient one")
    details = {}
    if np.any(np.linalg.eigvalsh((V + V.T) / 2) < 0):
        details["note"] = "the covariance difference is not positive definite; the statistic uses a generalised inverse"
    return TestResult("Hausman", "the efficient estimator is consistent (e.g. random effects are uncorrelated with the regressors)",
                      stat, df, float(sps.chi2.sf(stat, df)), "chi2",
                      details={"coefficients": ", ".join(common), "variance": sigma, **details})
