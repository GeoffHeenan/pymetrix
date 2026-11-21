"""A custom estimator: ridge regression, written with pymetrix.toolkit.

Once defined (and registered), it works like a built-in method:
``Equation(...).fit(data, method=MyRidge(alpha=5))``, the Estimate dialog's
method list, scripts, projects, group loops and side-by-side tables.
(pymetrix's own ``Ridge``, in the scikit-learn plugin, also chooses its penalty by
cross-validation; this one shows how a method is written.)

Run it from the repository folder:  python examples/custom_estimator.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np

from pymetrix import Dataset, Equation, toolkit
from pymetrix.forms import Widget


@dataclass(frozen=True)
class MyRidge:
    """Ridge regression: least squares with a penalty on the size of the coefficients (not the constant).

    The penalty ``alpha`` shrinks the coefficients towards zero, trading a little bias for less variance;
    it helps when regressors are nearly collinear. Regressors are standardised before shrinking, so the
    penalty treats them alike whatever their units.
    """

    label: ClassVar[str] = "My ridge"  # the name in the Estimate dialog
    kind: ClassVar[str] = "ls"  # a least-squares-type method
    topics: ClassVar[tuple[str, ...]] = ("least-squares",)

    alpha: Annotated[float, Widget("number", min=0, max=1e6, step=1, help="penalty on the coefficients; 0 = least squares")] = 1.0
    cov: Annotated[Literal["classic", "white"], Widget("choice", help="covariance of the (shrunk) coefficients")] = "classic"

    def estimate(self, eq, data, *, span=None, where=None, weights=None, missing="drop"):
        d = toolkit.design(eq, data, span=span, where=where, weights=weights, missing=missing)
        X, y = d.X, d.y
        penalised = np.array([name != "const" for name in d.names])
        scale = np.where(penalised, X.std(axis=0), 1.0)
        scale[scale == 0] = 1.0
        Xs = X / scale
        P = self.alpha * np.diag(penalised.astype(float))
        A = Xs.T @ Xs + P
        beta = np.linalg.solve(A, Xs.T @ y) / scale  # back to the original units
        u = y - X @ beta
        # covariance of the ridge estimator: A^-1 X' Omega X A^-1, in the original units
        Ainv = np.linalg.inv(A)
        meat = Xs.T @ Xs * (u @ u / d.df_resid) if self.cov == "classic" else (Xs * u[:, None]).T @ (Xs * u[:, None])
        V = (Ainv @ meat @ Ainv) / np.outer(scale, scale)
        return toolkit.result(eq, data, d, self, beta, V, u, extra={"alpha": self.alpha})


toolkit.register("methods", MyRidge)  # now in the Estimate dialog's method list


if __name__ == "__main__":
    data = Dataset.read("examples/sample_macro.parquet")
    eq = Equation("c ~ y + y[-1] + i")
    for alpha in (0.0, 10.0, 100.0):
        fit = eq.fit(data, method=MyRidge(alpha=alpha))
        print(f"alpha = {alpha:g}:", dict(zip(fit.params.names, np.round(fit.params.values, 4))))
    print(eq.fit(data, method=MyRidge(alpha=10)).summary())
