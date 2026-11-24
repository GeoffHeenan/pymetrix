"""The EViews plugin's tests: skipped where the plugin is not installed (the core-only run)."""

import importlib.util

collect_ignore_glob = [] if importlib.util.find_spec("pymetrix_eviews") is not None else ["test_*.py"]
