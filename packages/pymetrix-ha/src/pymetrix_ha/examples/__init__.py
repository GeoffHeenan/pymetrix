"""Example models, matched to sequence-jacobian's published notebooks.

- ``krusell_smith()``: Krusell and Smith's (1998) model as the package's Krusell-Smith notebook calibrates it (seven
  income states, 500 asset points, a capital share of 0.11), with a TFP shock ``Z``.
"""

from __future__ import annotations

from typing import Any


def krusell_smith(rho: float = 0.8, sd: float = 0.01, horizon: int = 300, **calibration: Any) -> Any:
    """The Krusell-Smith model of sequence-jacobian's notebook as an ``HAModel``: households that save in capital
    against uninsurable income risk, a Cobb-Douglas firm, and the asset market clearing. The steady state finds the
    discount factor ``beta``, TFP ``Z`` and capital ``K`` for an interest rate of 1%, output 1 and a cleared market;
    the dynamics solve for the path of ``K``. ``rho`` and ``sd``: the TFP shock's AR(1) path (the notebook's 1%
    shock decaying at 0.8). Keyword arguments change the calibration (``nA=100`` for a smaller asset grid)."""
    from ..model import AR1, HAModel
    from . import _krusell_smith as ks

    unknown = set(calibration) - set(ks.CALIBRATION)
    if unknown:
        from pymetrix.errors import SpecificationError

        raise SpecificationError(f"krusell_smith: no calibration value {sorted(unknown)}; it has {', '.join(ks.CALIBRATION)}")
    return HAModel([ks.household_ext, ks.firm, ks.mkt_clearing], calibration={**ks.CALIBRATION, **calibration},
                   steady_state=ks.STEADY_STATE, equilibrium=ks.EQUILIBRIUM, shocks={"Z": AR1(rho=rho, sd=sd)},
                   horizon=horizon, name="Krusell-Smith")


__all__ = ["krusell_smith"]
