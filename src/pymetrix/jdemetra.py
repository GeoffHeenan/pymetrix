"""Moved: JDemetra+ (TRAMO-SEATS) is in the ``pymetrix-seasonal`` plugin, as ``pymetrix_seasonal.jdemetra``.

This module forwards to it when the plugin is installed, so ``pymetrix.jdemetra.install()`` keeps working:

    pip install "pymetrix-seasonal[jdemetra]"
    python -c "import pymetrix_seasonal.jdemetra as j; j.install()"
"""

from __future__ import annotations

from typing import Any


def _plugin() -> Any:
    try:
        from pymetrix_seasonal import jdemetra
    except ImportError:
        from .errors import PluginUnavailable

        raise PluginUnavailable('pymetrix-seasonal: TRAMO-SEATS is in the seasonal-adjustment plugin: '
                                'pip install "pymetrix-seasonal[jdemetra]"') from None
    return jdemetra


def __getattr__(name: str) -> Any:
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(_plugin(), name)
