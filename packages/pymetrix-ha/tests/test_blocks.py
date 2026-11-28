"""Blocks and the model's checks: sequence-jacobian reads outputs from source text, so blocks run by exec and blocks
that return an expression are caught with a message that says what to write; the options are the library's own
defaults, passed explicitly; the help topic and the plugin's record."""

import inspect

import numpy as np
import pytest

from pymetrix.errors import SpecificationError

import pymetrix_ha
from pymetrix_ha import AR1, BlockError, HAModel, HAOptions, blocks_from_source
from pymetrix_ha.examples import _krusell_smith as nb
from pymetrix_ha.examples import krusell_smith

FIRM = '''
import pymetrix_ha

@pymetrix_ha.simple
def firm(K, L, Z, alpha, delta):
    r = alpha * Z * (K(-1) / L) ** (alpha - 1) - delta
    w = (1 - alpha) * Z * (K(-1) / L) ** alpha
    Y = Z * K(-1) ** alpha * L ** (1 - alpha)
    return r, w, Y


@pymetrix_ha.simple
def mkt_clearing(K, A, Y, C, delta):
    asset_mkt = A - K
    goods_mkt = Y - C - delta * K
    return asset_mkt, goods_mkt
'''


def _small(blocks):
    return HAModel(blocks, calibration={**nb.CALIBRATION, "nA": 60}, steady_state=nb.STEADY_STATE,
                   equilibrium=nb.EQUILIBRIUM, shocks={"Z": AR1(0.8, 0.01)}, horizon=40)


def test_blocks_run_by_exec_fail_in_the_library_with_a_message_that_says_what_to_do():
    import sequence_jacobian as sj

    scope = {"sj": sj}
    with pytest.raises(OSError, match="source"):
        exec("@sj.simple\ndef firm(K, alpha):\n    r = alpha * K\n    return r\n", scope)
    scope = {"pymetrix_ha": pymetrix_ha}
    with pytest.raises(BlockError, match="blocks_from_source") as info:
        exec("@pymetrix_ha.simple\ndef firm(K, alpha):\n    r = alpha * K\n    return r\n", scope)
    assert "'firm'" in str(info.value)


def test_blocks_from_source_work_as_if_they_were_in_a_file():
    """The same blocks from text and from a file give the same steady state, G and transition."""
    ns = blocks_from_source(FIRM, name="ks")
    assert inspect.getsource(ns["firm"].f.f).lstrip().startswith("@pymetrix_ha.simple")
    from_text = _small([nb.household_ext, ns["firm"], ns["mkt_clearing"]])
    from_file = _small([nb.household_ext, nb.firm, nb.mkt_clearing])
    a, b = from_text.steady_state(), from_file.steady_state()
    assert a.values == b.values
    assert np.array_equal(from_text.linearize(a).jacobian("K", "Z"), from_file.linearize(b).jacobian("K", "Z"))
    assert [blk["source"] for blk in a.provenance.spec["blocks"]][1]  # text blocks are fingerprinted too


def test_a_returned_expression_is_refused_with_the_block_named():
    ns = blocks_from_source("import sequence_jacobian as sj\n\n@sj.simple\ndef euler(r, c):\n    return (1 + r) * c ** -1\n")
    with pytest.raises(BlockError, match=r"block 'euler'.*not names.*return the\s+names on one line"):
        pymetrix_ha.check_blocks([ns["euler"]])
    with pytest.raises(BlockError, match="'euler'"):
        blocks_from_source("import pymetrix_ha\n\n@pymetrix_ha.simple\ndef euler(r, c):\n    return (1 + r) * c ** -1\n")
    with pytest.raises(BlockError, match="'euler'"):
        HAModel([ns["euler"]], calibration={"r": 0.01, "c": 1.0}, steady_state={"unknowns": {}, "targets": {}},
                equilibrium={"unknowns": [], "targets": []})


def test_inputs_targets_and_shocks_are_checked_when_the_model_is_made():
    calibration = {k: v for k, v in nb.CALIBRATION.items() if k != "delta"}
    with pytest.raises(BlockError, match=r"block 'firm' needs 'delta'"):
        HAModel([nb.household_ext, nb.firm, nb.mkt_clearing], calibration=calibration, steady_state=nb.STEADY_STATE,
                equilibrium=nb.EQUILIBRIUM, shocks={"Z": AR1(0.8, 0.01)})
    with pytest.raises(SpecificationError, match="'asset_market'"):
        HAModel([nb.household_ext, nb.firm, nb.mkt_clearing], calibration=nb.CALIBRATION, steady_state=nb.STEADY_STATE,
                equilibrium={"unknowns": ["K"], "targets": ["asset_market"]}, shocks={"Z": AR1(0.8, 0.01)})
    with pytest.raises(SpecificationError, match="rho is between -1 and 1"):
        AR1(1.2, 0.01)
    with pytest.raises(BlockError, match="no block 'households'"):
        krusell_smith(nA=60).jacobian("households", ["r"])


def test_the_options_are_the_librarys_own_defaults():
    """HAOptions passes sequence-jacobian 1.0.0's defaults explicitly: read here from the library's signatures, so
    that a release that changes one fails this test instead of changing results silently."""
    from sequence_jacobian.blocks.block import Block
    from sequence_jacobian.blocks.het_block import HetBlock

    o = HAOptions()
    ss_opts = Block.solve_steady_state_options
    het_ss = {k: p.default for k, p in inspect.signature(HetBlock._steady_state).parameters.items() if p.default is not p.empty}
    het_jac = {k: p.default for k, p in inspect.signature(HetBlock._jacobian).parameters.items() if p.default is not p.empty}
    nonlin = Block.solve_impulse_nonlinear_options
    assert (o.ttol, o.ctol) == (ss_opts["ttol"], ss_opts["ctol"])
    assert (o.backward_tol, o.backward_maxit, o.forward_tol, o.forward_maxit) == (
        het_ss["backward_tol"], het_ss["backward_maxit"], het_ss["forward_tol"], het_ss["forward_maxit"])
    assert (o.h, o.twosided) == (het_jac["h"], het_jac["twosided"])
    assert (o.tol, o.maxit) == (nonlin["tol"], nonlin["maxit"])
    assert o.steady_solver({"beta": (0.9, 0.99)}) == "brentq" and o.steady_solver({"beta": 0.98, "K": 3.0}) == "hybr"


def test_the_plugin_record_and_help_topic():
    from pymetrix import help as hp

    assert pymetrix_ha.PLUGIN.name == "pymetrix-ha" and pymetrix_ha.__version__ == "0.1.0a1"
    assert pymetrix_ha.PLUGIN.available and set(pymetrix_ha.PLUGIN.requires) == {"sequence_jacobian", "numba"}
    topic = hp.topic("heterogeneous-agents")
    assert "sequence-jacobian" in topic.text and "blocks_from_source" in topic.text
    cited = " ".join(topic.references)
    for author in ("Auclert, A., Bardóczy, B., Rognlie, M. and Straub, L. (2021)", "Krusell, P. and Smith, A. A. (1998)",
                   "Young, E. R. (2010)", "Carroll, C. D. (2006)"):
        assert author in cited
    assert "doi:" not in cited  # none typed from memory: added once scripts/check_references.py confirms them
