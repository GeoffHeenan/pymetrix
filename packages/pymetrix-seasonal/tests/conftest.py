"""The seasonal plugin's tests: skipped where it is not installed (the core-only run)."""

import importlib.util

collect_ignore_glob = [] if importlib.util.find_spec("pymetrix_seasonal") is not None else ["test_*.py"]
