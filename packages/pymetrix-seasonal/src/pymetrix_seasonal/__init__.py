"""First-party plugin: seasonal adjustment by external programs (``pymetrix-seasonal``,
``pip install pymetrix-seasonal``; ``[jdemetra]`` adds JPype for TRAMO-SEATS).

- ``X13``: the US Census Bureau's X-13ARIMA-SEATS, its ``x13as`` program run on a specification written here.
- ``TramoSeats``: TRAMO-SEATS as Eurostat's JDemetra+ 3 computes it, in Java (``pymetrix_seasonal.jdemetra``).

Both register as seasonal-adjustment methods, so pymetrix's Seasonal adjustment dialog and
``Dataset.seasonal_adjust`` offer them beside its own STL and classical decomposition, and adjusted series keep
updating as before. They live here, not in pymetrix, because they need programs outside Python (x13as; Java 21 and
JDemetra+'s libraries). ``from pymetrix_seasonal import X13``; ``from pymetrix import X13`` works too when this
package is installed.
"""

from __future__ import annotations

from pymetrix.extension import PluginInfo, plugin_context
from pymetrix.extension import register as add

from ._version import __version__

PLUGIN = PluginInfo(
    name="pymetrix-seasonal",
    version=__version__,
    license="BSD-3-Clause",
    maintainer="Geoffrey Heenan",
    description="Seasonal adjustment by X-13ARIMA-SEATS (the x13as program) and TRAMO-SEATS (JDemetra+, in Java)",
    # no requires: X-13 needs only its program; TRAMO-SEATS says what it lacks (JPype, Java, the jars) when used
    install='pip install "pymetrix-seasonal[jdemetra]"',
)

from .tramo import TramoSeats  # noqa: E402
from .x13 import X13, run_x13, x13_notes, x13_program  # noqa: E402

with plugin_context(PLUGIN):
    for _cls in (X13, TramoSeats):
        add("seasonal", _cls)


def register() -> None:
    """The entry point: importing the package has registered both methods."""


__all__ = ["PLUGIN", "TramoSeats", "X13", "__version__", "register", "run_x13", "x13_notes", "x13_program"]
