"""``import pymetrix`` loads numpy, pandas and scipy (which it needs) but none of the heavy optional libraries: they
are imported where they are used, so a script that never plots, opens the interface or calls a backend never pays
for them. Run in a fresh interpreter, since the test session has imported them already."""

import os
import subprocess
import sys

OPTIONAL = ("statsmodels", "matplotlib", "linearmodels", "sklearn", "arch", "PySide6", "jpype", "sympy", "numba")


def test_import_leaves_optional_libraries_unloaded():
    code = ("import sys, pymetrix\n"
            f"print(','.join(m for m in {OPTIONAL!r} if m in sys.modules))")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
    assert out.stdout.strip() == ""
