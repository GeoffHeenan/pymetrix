"""The heterogeneous-agent plugin's tests: skipped where the plugin or sequence-jacobian (with numba) is not installed
(the core-only run, and any Python numba does not support yet)."""

import importlib.util

_ready = all(importlib.util.find_spec(m) is not None for m in ("pymetrix_ha", "sequence_jacobian", "numba"))
collect_ignore_glob = [] if _ready else ["test_*.py"]
