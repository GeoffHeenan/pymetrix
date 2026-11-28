"""The Krusell-Smith model of sequence-jacobian's Krusell-Smith notebook (``notebooks/krusell_smith.ipynb`` at the
repository's commit eb4ac4c): its blocks as the notebook writes them, so that the plugin's results can be checked
against the numbers it prints. The blocks are adapted from sequence-jacobian (MIT licence; THIRD-PARTY-NOTICES.md).

The asset grid is ``grids.agrid``, as in the notebook; the newer ``grids.asset_grid`` places the points differently
and gives entries of the household Jacobian up to 0.3% apart.
"""

from __future__ import annotations

import numpy as np
from sequence_jacobian import grids, het, interpolate, misc, simple


def household_init(a_grid, e_grid, r, w, eis):
    coh = (1 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis]
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)
    return Va


@het(exogenous="Pi", policy="a", backward="Va", backward_init=household_init)
def household(Va_p, a_grid, e_grid, r, w, beta, eis):
    """One backward step of the endogenous grid method (Carroll 2006) for a household with CRRA utility: from next
    period's expected marginal value of assets to today's, with the asset and consumption policies."""
    uc_nextgrid = beta * Va_p
    c_nextgrid = uc_nextgrid ** (-eis)
    coh = (1 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis]
    a = interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    misc.setmin(a, a_grid[0])
    c = coh - a
    Va = (1 + r) * c ** (-1 / eis)
    return Va, a, c


def make_grid(rho_e, sd_e, nE, amin, amax, nA):
    """The income process (Rouwenhorst's discretisation of an AR(1) in log income) and the asset grid."""
    e_grid, _, Pi = grids.markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    a_grid = grids.agrid(amin=amin, amax=amax, n=nA)
    return e_grid, Pi, a_grid


household_ext = household.add_hetinputs([make_grid])


@simple
def firm(K, L, Z, alpha, delta):
    r = alpha * Z * (K(-1) / L) ** (alpha - 1) - delta
    w = (1 - alpha) * Z * (K(-1) / L) ** alpha
    Y = Z * K(-1) ** alpha * L ** (1 - alpha)
    return r, w, Y


# goods_mkt as the notebook writes it, Y - C - delta * K, holds in the steady state only (along a path investment is
# K - (1 - delta) * K(-1)); it is not a target, and the notebook's numbers depend only on asset_mkt
@simple
def mkt_clearing(K, A, Y, C, delta):
    asset_mkt = A - K
    goods_mkt = Y - C - delta * K
    return asset_mkt, goods_mkt


CALIBRATION = {"eis": 1, "delta": 0.025, "alpha": 0.11, "rho_e": 0.966, "sd_e": 0.5, "L": 1.0,
               "nE": 7, "nA": 500, "amin": 0, "amax": 200}
STEADY_STATE = {"unknowns": {"beta": 0.98, "Z": 0.85, "K": 3.0}, "targets": {"r": 0.01, "Y": 1.0, "asset_mkt": 0.0},
                "solver": "hybr"}
EQUILIBRIUM = {"unknowns": ["K"], "targets": ["asset_mkt"]}
