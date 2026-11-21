"""A missing optional library is explained by the extra that installs it; a missing core one as a broken install."""

import importlib

import pytest

from pymetrix import adapters


@pytest.mark.parametrize(("module", "hint"), [("sklearn.linear_model", "pip install pymetrix[ml]"), ("mcp.server.mcpserver", "pip install pymetrix[mcp]"), ("arch.bootstrap", "pip install pymetrix[arch]"),
                                              ("linearmodels.iv", "pip install pymetrix[panel]"), ("statsmodels.api", "which pymetrix requires")])
def test_require_names_the_install(module, hint, monkeypatch):
    def missing(name, *args, **kw):
        raise ModuleNotFoundError(f"No module named {name!r}")

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(ImportError, match=hint.replace("[", r"\[").replace("]", r"\]")):
        adapters.require(module)


def test_every_plugin_install_hint_is_an_extra():
    """Each optional library the bundled plugins wrap maps to an extra, so its error names a real install command; a
    library no extra can install (PSETS is not on PyPI) needs the plugin's own ``install`` command instead."""
    from pymetrix.plugins import plugins

    for info in plugins():
        for library in info.requires:
            assert library in adapters.EXTRAS or info.install, (
                f"{info.name} needs {library}, which adapters.EXTRAS does not map to an extra and the plugin gives no install command")
