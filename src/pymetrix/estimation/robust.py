"""Robust regression by M-estimation (Huber 1964, 1973), with Huber's or Tukey's bisquare norm.

    Equation("y ~ x").fit(data, method=RobustLS())                    # Huber's norm, scale estimated
    Equation("y ~ x").fit(data, method=RobustLS(norm="bisquare"))     # Tukey's bisquare: outliers get no weight

The coefficients minimise ``sum rho((y - X b) / s)``: ``rho`` grows like a square for small residuals and more slowly
(Huber: linearly) or not at all (bisquare) for large ones, so outlying observations of the dependent variable pull less.
They are computed by iteratively reweighted least squares from the least-squares start: each round weights every
observation by ``psi(z) / z`` at its scaled residual ``z = u / s`` and re-estimates by weighted least squares; the scale
``s`` is re-estimated each round as the median absolute residual divided by 0.6745 (the MAD, consistent for the standard
deviation of normal errors), unless it is given. The tuning constants (1.345 for Huber, 4.685 for the bisquare) give
95% efficiency when the errors are normal.

The covariance is Huber's H1, as statsmodels' RLM computes it::

    K^2 * [sum psi(z)^2 / (n - k)] * s^2 / [mean psi'(z)]^2 * (X'X)^-1,   K = 1 + (k / n) var(psi'(z)) / mean(psi'(z))^2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd

from ..errors import SpecificationError
from ..forms import Widget

if TYPE_CHECKING:
    from ..data import Dataset
    from .equation import Equation
    from .fitted import FittedEquation

__all__ = ["RobustLS"]

_TUNING = {"huber": 1.345, "bisquare": 4.685}
_MAD_NORMAL = 0.6744897501960817  # the standard normal's 75% quantile: MAD / this estimates sigma under normal errors


@dataclass(frozen=True)
class RobustLS:
    """Robust regression (M-estimation): Huber's norm or Tukey's bisquare, the scale estimated by the MAD.

    Large residuals are down-weighted (Huber) or, beyond the tuning constant, given no weight at all (bisquare), so
    a few outlying values of the dependent variable do not drive the fit. Huber's estimate is unique; the bisquare's
    objective is not convex, so it starts from least squares and finds the nearest solution, which is the usual
    practice. Neither protects against outlying regressors (leverage points). With normal errors the default tuning
    constants lose about 5% efficiency against least squares. The standard errors (Huber's H1) assume the errors are
    independent and symmetric; ``fit.extra["robust_weights"]`` gives each observation's final weight, and those near
    zero are the outliers.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import Dataset, Equation, RobustLS

        card = Dataset.example("card")
        eq = Equation("lwage ~ educ + exper + black")
        print(eq.fit(card, method=RobustLS()).summary())                  # Huber's norm: large residuals count less
        fit = eq.fit(card, method=RobustLS(norm="bisquare"))              # Tukey's bisquare: outliers get no weight
        fit.extra["robust_weights"].sort_values().head()                  # the observations weighted least
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("coef, std err, t", ("M-estimates with Huber's H1 standard errors (as statsmodels' RLM), which assume "
                              "independent errors symmetric about zero")),
        ("Robust regression note", "the norm and its tuning constant, the scale and how it was found, and the iterations"),
        ("S.E. of regression, criteria", ("from the residuals of the robust fit, as for least squares; the criteria are "
                                          "a Gaussian quasi-likelihood, not the M-estimator's objective")),
        ("fit.extra[\"robust_weights\"]", "each observation's final weight, from 1 down to 0 for the outliers"),
        ("fit.stats", "scale, iterations, ssr, se_regression, loglike, aic, bic, hq, nobs and df_resid"),
    )

    label: ClassVar[str] = "Robust regression (M)"
    kind: ClassVar[str] = "robust"
    topics: ClassVar[tuple[str, ...]] = ("robust-regression",)
    references: ClassVar[tuple[str, ...]] = (
        "Huber, P. J. (1964). Robust estimation of a location parameter. Annals of Mathematical Statistics 35(1), 73-101. doi:10.1214/aoms/1177703732",
        "Huber, P. J. (1973). Robust regression: asymptotics, conjectures and Monte Carlo. Annals of Statistics 1(5), 799-821. doi:10.1214/aos/1176342503",
        "Beaton, A. E. and Tukey, J. W. (1974). The fitting of power series, meaning polynomials, illustrated on band-spectroscopic data. Technometrics 16(2), 147-185. doi:10.1080/00401706.1974.10489171",
        "Holland, P. W. and Welsch, R. E. (1977). Robust regression using iteratively reweighted least-squares. Communications in Statistics - Theory and Methods 6(9), 813-827. doi:10.1080/03610927708827533",
    )

    norm: Annotated[Literal["huber", "bisquare"], Widget("choice", help="huber: large residuals weighted down; bisquare (Tukey): "
                                                                         "residuals beyond the tuning constant get no weight")] = "huber"
    tuning: Annotated[float | None, Widget("number", min=0.1, max=20, step=0.1, advanced=True,
                                           help="the norm's tuning constant, in scaled residuals; empty = 1.345 (Huber) or "
                                                "4.685 (bisquare), 95% efficient with normal errors; smaller is more robust")] = None
    scale: Annotated[float | None, Widget("number", min=0, max=1e12, advanced=True,
                                          help="the errors' scale, held fixed; empty = estimated by the MAD of the residuals "
                                               "and updated each iteration")] = None
    max_iter: Annotated[int, Widget("number", min=1, max=10_000, advanced=True, help="iterations of reweighted least squares")] = 200
    tol: Annotated[float, Widget("number", min=1e-15, max=1e-2, advanced=True,
                                 help="convergence: the largest change in the coefficients and the scale, relative to 1 + their size")] = 1e-10

    def __post_init__(self) -> None:
        if self.norm not in _TUNING:
            raise SpecificationError(f"norm is 'huber' or 'bisquare', not {self.norm!r}")
        if self.tuning is not None and not self.tuning > 0:
            raise SpecificationError("the tuning constant must be positive")
        if self.scale is not None and not self.scale > 0:
            raise SpecificationError("a fixed scale must be positive (empty: estimated)")

    @property
    def constant(self) -> float:
        """The tuning constant used."""
        return float(self.tuning) if self.tuning is not None else _TUNING[self.norm]

    # the norm: weights psi(z)/z, psi and psi'

    def _weights(self, z: np.ndarray) -> np.ndarray:
        c, a = self.constant, np.abs(z)
        if self.norm == "huber":
            return np.where(a <= c, 1.0, c / np.where(a <= c, 1.0, a))
        return np.where(a <= c, (1 - (z / c) ** 2) ** 2, 0.0)

    def _psi(self, z: np.ndarray) -> np.ndarray:
        c = self.constant
        if self.norm == "huber":
            return np.clip(z, -c, c)
        return np.where(np.abs(z) <= c, z * (1 - (z / c) ** 2) ** 2, 0.0)

    def _psi_deriv(self, z: np.ndarray) -> np.ndarray:
        c = self.constant
        inside = np.abs(z) <= c
        if self.norm == "huber":
            return inside.astype(float)
        u = (z / c) ** 2
        return np.where(inside, (1 - u) ** 2 - 4 * u * (1 - u), 0.0)

    def estimate(self, eq: Equation, data: Dataset, *, span: Any = None, where: str | None = None, weights: str | None = None,
                 missing: str = "drop") -> FittedEquation:
        from .design import linear_design, ls_stats, provenance
        from .fitted import FittedEquation
        from .methods import _check_rank, _require_gaussian
        from .params import point

        _require_gaussian(eq, "Robust regression")
        if weights:
            raise SpecificationError("robust regression makes its own weights: leave Weights empty")
        if eq.form != "linear" or eq.fixed_effects:
            raise SpecificationError("robust regression needs a linear-form equation without fe()")
        d = linear_design(eq, data, span, where, None, missing)
        y, X = d.y, d.X
        n, k = X.shape
        if n <= k:
            raise SpecificationError(f"robust regression needs more observations ({n}) than coefficients ({k})")
        _check_rank(X, d.names, "regressors")
        beta, iterations, converged, s = self._irls(y, X)
        u = y - X @ beta
        z = u / s
        dpsi = self._psi_deriv(z)
        m = dpsi.mean()
        if m <= 0:
            raise SpecificationError("no observation lies within the tuning constant: the covariance cannot be computed; "
                                     "raise the tuning constant or use Huber's norm")
        K = 1 + k / n * dpsi.var() / m**2
        cov = K**2 * (np.sum(self._psi(z) ** 2) / (n - k) * s**2) / m**2 * np.linalg.inv(X.T @ X)
        stats = ls_stats(y, u, None, k, n - k, eq.intercept, data.kind == "time_series", False)
        stats.pop("r2", None), stats.pop("r2_adj", None)
        stats.update({"scale": float(s), "iterations": float(iterations), "df_resid": float(n - k)})
        how = "fixed" if self.scale is not None else "by the MAD of the residuals, updated each iteration"
        notes = [f"Robust regression: {'Huber' if self.norm == 'huber' else 'Tukey bisquare'} norm, tuning constant {self.constant:g}; "
                 f"scale {s:.6g} ({how}); {iterations} iterations" + ("" if converged else " (did not converge: raise max_iter)"),
                 "Standard errors: Huber's H1"]
        weights_used = pd.Series(self._weights(z), index=d.index, name="robust weight")
        return FittedEquation(eq, point(d.names, beta, cov), self,
                              resid=pd.Series(u, index=d.index, name="resid"), fitted=pd.Series(d.y_observed - u, index=d.index, name="fitted"),
                              stats=stats, provenance=provenance(eq, self, data, d.index, d.dropped),
                              design=pd.DataFrame(X, index=d.index, columns=d.names),
                              extra={"notes": notes, "converged": converged, "robust_weights": weights_used})

    def _irls(self, y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, int, bool, float]:
        """Iteratively reweighted least squares from the least-squares start: (coefficients, iterations, converged, scale)."""
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        fixed = self.scale is not None
        s = float(self.scale) if fixed else _mad(y - X @ beta)
        for iteration in range(1, self.max_iter + 1):
            if s <= 0:
                raise SpecificationError("the scale of the residuals is zero: the fit is exact for at least half the "
                                         "observations, so robust weights cannot be formed")
            w = np.sqrt(self._weights((y - X @ beta) / s))
            new = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
            s_new = s if fixed else _mad(y - X @ new)
            done = (np.max(np.abs(new - beta)) <= self.tol * (1 + np.max(np.abs(new)))
                    and abs(s_new - s) <= self.tol * (1 + s))
            beta, s = new, s_new
            if done:
                return beta, iteration, True, s
        return beta, self.max_iter, False, s


def _mad(u: np.ndarray) -> float:
    """The median absolute residual over the normal's 75% quantile (centred at zero, as the residuals are)."""
    return float(np.median(np.abs(u)) / _MAD_NORMAL)


def _register() -> None:
    from ..plugins import register

    register("methods", RobustLS, name="robust")


_register()
