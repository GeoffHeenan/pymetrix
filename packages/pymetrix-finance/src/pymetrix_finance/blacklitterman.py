"""The Black-Litterman result: the prior (implied equilibrium) and posterior expected returns, the posterior
covariance, the views with their uncertainties, and the weights the posterior makes optimal, labelled by asset.

    bl = black_litterman(data.cov(["a", "b", "c"]), [0.5, 0.3, 0.2], views=[[1, -1, 0]], q=[0.002])
    bl.summary(); bl.views_table(); bl.plot()       # by asset: prior, posterior, weights; the views; prior vs posterior
    mu, sigma = bl                                  # the posterior mean and covariance as arrays, as before

The optimal weights are the unconstrained mean-variance weights of an investor with the market's risk aversion,
w = (delta Sigma_post)^-1 mu_post (He and Litterman 1999): with no views they are the market weights scaled by
1 / (1 + tau), and a view moves the weights only of the assets in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import pandas as pd

from pymetrix.errors import SpecificationError
from pymetrix.output import Table

__all__ = ["BlackLittermanPosterior", "view_label"]


def view_label(row: np.ndarray, assets: list[str]) -> str:
    """A row of view weights as written in the dialog: ``a - b``, ``0.5 c + 0.5 a``."""
    out = ""
    for w, a in zip(row, assets):
        if w == 0:
            continue
        size = abs(float(w))
        term = a if size == 1 else f"{size:g} {a}"
        out += (f"- {term}" if w < 0 else term) if not out else (f" - {term}" if w < 0 else f" + {term}")
    return out or "0"


@dataclass(frozen=True, eq=False)
class BlackLittermanPosterior:
    """Black-Litterman expected returns: the prior implied by the market weights, the views, and the posterior. Unpacks
    as ``mu, sigma`` (the posterior mean and covariance, arrays in the assets' order), as the function's tuple did."""

    assets: tuple[str, ...]
    prior: pd.Series  # the implied equilibrium returns, pi = delta S w_m
    posterior_mean: pd.Series
    posterior_cov: pd.DataFrame
    cov: pd.DataFrame = field(repr=False)  # the returns' covariance S
    market_weights: pd.Series = field(repr=False)
    views: pd.DataFrame | None = field(default=None, repr=False)  # P, one row a view (labelled as written), one column an asset
    q: pd.Series | None = field(default=None, repr=False)  # each view's value
    omega: pd.DataFrame | None = field(default=None, repr=False)  # the views' covariance (their uncertainty)
    omega_default: bool = True  # omega = diag(P tau S P') (He and Litterman), not given
    delta: float = 2.5
    tau: float = 0.05
    provenance: Any = field(default=None, repr=False)

    stage = "finance"

    # -- the old tuple ---------------------------------------------------------------------------------------------
    def _pair(self) -> tuple[np.ndarray, np.ndarray]:
        return self.posterior_mean.to_numpy(), self.posterior_cov.to_numpy()

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self._pair())

    def __getitem__(self, i: int) -> np.ndarray:
        return self._pair()[i]

    def __len__(self) -> int:
        return 2

    # -- numbers ---------------------------------------------------------------------------------------------------
    @property
    def implied_returns(self) -> pd.Series:
        return self.prior

    @property
    def confidences(self) -> pd.Series | None:
        """Each view's variance (the diagonal of omega): smaller is more confident."""
        return None if self.omega is None else pd.Series(np.diag(self.omega.to_numpy()), index=self.omega.index, name="variance")

    @property
    def weights(self) -> pd.Series:
        """The unconstrained mean-variance weights for the posterior, (delta Sigma_post)^-1 mu_post."""
        if self.delta <= 0:
            raise SpecificationError("the optimal weights need a positive risk aversion delta")
        w = np.linalg.solve(self.delta * self.posterior_cov.to_numpy(), self.posterior_mean.to_numpy())
        return pd.Series(w, index=self.posterior_mean.index, name="optimal weight")

    def table_frame(self) -> pd.DataFrame:
        """By asset: market weight, prior (implied) return, posterior return, their difference, posterior standard
        deviation and optimal weight."""
        out = pd.DataFrame({"market weight": self.market_weights, "prior return": self.prior, "posterior return": self.posterior_mean,
                            "change": self.posterior_mean - self.prior, "posterior sd": np.sqrt(np.diag(self.posterior_cov.to_numpy()))},
                           index=self.prior.index)
        if self.delta > 0:
            out["optimal weight"] = self.weights
        return out

    def views_frame(self) -> pd.DataFrame:
        """By view: its value q, variance (omega's diagonal) and standard deviation, and the view's value under the
        prior (P pi) and the posterior (P mu)."""
        if self.views is None or self.q is None or self.omega is None:
            return pd.DataFrame(columns=["q", "variance", "sd", "prior", "posterior"], index=pd.Index([], name="view"))
        P = self.views.to_numpy()
        var = np.diag(self.omega.to_numpy())
        return pd.DataFrame({"q": self.q.to_numpy(), "variance": var, "sd": np.sqrt(var), "prior": P @ self.prior.to_numpy(),
                             "posterior": P @ self.posterior_mean.to_numpy()}, index=self.views.index)

    # -- views -----------------------------------------------------------------------------------------------------
    def summary(self) -> Table:
        f = self.table_frame()
        n = 0 if self.views is None else len(self.views)
        notes = [f"Prior: the returns implied by the market weights, pi = delta S w with delta = {self.delta:g}; tau = {self.tau:g}.",
                 (f"{n} view(s), " + ("each as uncertain as the prior on it (omega = diag(P tau S P'), He and Litterman)."
                                      if self.omega_default else "with the variances given (omega).")) if n else
                 "No views: the posterior mean is the prior, and the covariance S (1 + tau).",
                 "Optimal weight: unconstrained mean-variance weights for the posterior, (delta Sigma)^-1 mu."]
        return Table(list(f.columns), f.to_numpy(float).tolist(), index=list(f.index), index_name="asset", title="Black-Litterman",
                     notes=tuple(notes))

    def views_table(self) -> Table:
        f = self.views_frame()
        return Table(list(f.columns), f.to_numpy(float).tolist(), index=[str(i) for i in f.index], index_name="view", title="Views",
                     notes=("q: the view's value; variance and sd: its uncertainty (omega); prior and posterior: the view's value "
                            "under each mean.",))

    def plot(self) -> Any:
        from pymetrix.figures import Bars, Figure, Panel

        idx = pd.Index(list(self.assets), name="asset")
        prior = pd.Series(self.prior.to_numpy(), index=idx)
        post = pd.Series(self.posterior_mean.to_numpy(), index=idx)
        return Figure((Panel((Bars(prior, "prior (implied)"), Bars(post, "posterior")), title="Expected returns: prior and posterior"),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from pymetrix.extension import views_frame

            return views_frame(self, view)
        return self.table_frame()

    def __repr__(self) -> str:
        n = 0 if self.views is None else len(self.views)
        return f"<BlackLittermanPosterior of {', '.join(self.assets)}, {n} view(s): {dict(zip(self.assets, (round(float(v), 6) for v in self.posterior_mean)))}>"
