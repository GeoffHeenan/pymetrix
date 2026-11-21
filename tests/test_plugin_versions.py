"""Plugins' version requirements are enforced, not just their presence (review R3)."""

import numpy as np

from pymetrix.plugins import PluginInfo, installed_version, version_problem


def test_installed_versions_and_specifiers():
    assert installed_version("numpy") == np.__version__
    assert installed_version("no_such_library_xyz") is None
    assert version_problem("numpy", ">=1") is None
    assert version_problem("numpy", "") is None
    assert version_problem("numpy", ">=999") == f"numpy {np.__version__} is installed, but >=999 is needed"
    assert version_problem("numpy", "not a specifier") is None  # unreadable: not held against the plugin


def test_plugins_with_an_unsupported_version_are_unavailable():
    ok = PluginInfo("pymetrix-ok", "1.0", requires={"numpy": ">=1"})
    old = PluginInfo("pymetrix-old", "1.0", requires={"numpy": ">=999"}, install="pip install pymetrix-old")
    gone = PluginInfo("pymetrix-gone", "1.0", requires={"no_such_library_xyz": ">=1"})
    assert ok.available and ok.problem is None
    assert not old.available and old.missing == [] and old.unsupported
    assert old.problem == f"numpy {np.__version__} is installed, but >=999 is needed: pip install \"numpy>=999\""
    assert not gone.available and gone.problem == "needs no_such_library_xyz: pip install pymetrix-gone"
