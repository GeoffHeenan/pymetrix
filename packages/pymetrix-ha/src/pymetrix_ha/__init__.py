"""First-party plugin: heterogeneous-agent macro models (``pymetrix-ha``, ``pip install "pymetrix[ha]"``).

It wraps sequence-jacobian, the sequence-space Jacobian method of Auclert, Bardóczy, Rognlie and Straub (2021), and
returns pymetrix results: the steady state as a table, Jacobians as arrays by output and input, impulse responses as
``IRF`` and transitions as a ``Solution``, each with provenance (docs/design/heterogeneous-agents.md, phase 1).

    from pymetrix_ha.examples import krusell_smith
    ks = krusell_smith()
    ss = ks.steady_state()                 # ss.table()
    lin = ks.linearize(ss)                 # the general-equilibrium Jacobians G
    lin.irf(40).plot()                     # a pymetrix IRF
    path = ks.solve(steady=ss)             # the nonlinear perfect-foresight transition, a Solution

Models are written as sequence-jacobian's Python blocks (``simple``, ``het``). sequence-jacobian and numba are
imported when a model is made, not when the plugin loads.
"""

from __future__ import annotations

from pymetrix.extension import PluginInfo, plugin_context

from ._version import __version__

PLUGIN = PluginInfo(
    name="pymetrix-ha",
    version=__version__,
    license="BSD-3-Clause",
    maintainer="Geoffrey Heenan",
    description="Heterogeneous-agent macro models by the sequence-space Jacobian method (sequence-jacobian): steady "
                "states, Jacobians, impulse responses and nonlinear transitions",
    requires={"sequence_jacobian": ">=1.0,<2", "numba": ">=0.60"},  # Help -> Plugins shows their versions
    install='pip install "pymetrix[ha]"',
)

from .blocks import BlockError, blocks_from_source, check_blocks, het, simple  # noqa: E402
from .linear import HALinear, Jacobians  # noqa: E402
from .model import AR1, HAModel, HAOptions  # noqa: E402
from .nonlinear import HATransition  # noqa: E402
from .steady import HASteadyState  # noqa: E402

_registered = False


def register() -> None:
    """The entry point (``pymetrix.plugins``): the help topic. Importing the package registers it too; a second call
    does nothing."""
    global _registered
    if _registered:
        return
    _registered = True
    with plugin_context(PLUGIN):
        from . import app  # noqa: F401


register()

__all__ = ["AR1", "BlockError", "HALinear", "HAModel", "HAOptions", "HASteadyState", "HATransition", "Jacobians", "PLUGIN",
           "__version__", "blocks_from_source", "check_blocks", "het", "register", "simple"]
