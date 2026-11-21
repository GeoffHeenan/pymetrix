"""The finance plugin's tests: skipped where the plugin is not installed (the core-only run), and given the core
tests' gallery of example results."""

import importlib.util
import sys
from pathlib import Path

CORE_TESTS = Path(__file__).resolve().parents[3] / "tests"
if str(CORE_TESTS) not in sys.path:
    sys.path.insert(0, str(CORE_TESTS))

collect_ignore_glob = [] if importlib.util.find_spec("pymetrix_finance") is not None else ["test_*.py"]


if not collect_ignore_glob:
    import os
    import shutil

    import pytest

    _SERVED: dict = {}

    @pytest.fixture(autouse=True)
    def french_offline(tmp_path, monkeypatch):
        """No test reaches Kenneth French's data library: ``Dataset.example("french")`` downloads into a cache of the
        test's own (``PYMETRIX_HOME`` in tmp_path) from the simulated stand-in (french_stand_in.py), served in the
        library's file layout. ``PYMETRIX_FRENCH`` (the path of a french.parquet the plugin downloaded, such as
        ~/.pymetrix/cache/datasets/french.parquet) puts the real data in that cache instead. Returns the URLs fetched."""
        from pymetrix_finance.datasets import french

        import french_stand_in

        monkeypatch.setenv("PYMETRIX_HOME", str(tmp_path / "pymetrix_home"))
        if not _SERVED:
            _SERVED.update(french_stand_in.library())
        fetched: list[str] = []

        def get(url):
            fetched.append(url)
            return _SERVED[url]

        monkeypatch.setattr(french, "_get", get)
        real = os.environ.get("PYMETRIX_FRENCH")
        if real:
            french.cache_path().parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(real, french.cache_path())
        return fetched
