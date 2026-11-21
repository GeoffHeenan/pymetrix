"""The generated script of every demo session, run in a fresh Python process with no display, reproduces the session's
results, and loads no user-interface or plotting library: what the app writes is plain, headless Python."""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from pymetrix import demo, demos, statespace

SIGNATURE = textwrap.dedent('''
    def _signature(obj):
        """A few numbers that identify a result, whatever its kind."""
        import numpy as np
        out = {}
        params = getattr(obj, "params", None)
        if params is not None and hasattr(params, "values"):
            out["params"] = np.asarray(params.values, dtype=float).ravel().tolist()
        for name in ("ghs2", "ghx"):
            value = getattr(obj, name, None)
            if value is not None:
                out[name] = np.asarray(value, dtype=float).ravel().tolist()
        draws = getattr(obj, "draws", None)
        if isinstance(draws, np.ndarray):
            out["draws"] = draws.mean(axis=(0, 1)).tolist()
        frame = getattr(obj, "frame", None)
        if frame is not None and hasattr(frame, "select_dtypes"):
            out["frame"] = np.nan_to_num(frame.select_dtypes("number").to_numpy(dtype=float)).sum(axis=0).tolist()
        values = getattr(obj, "values", None)
        if isinstance(values, dict):
            out["values"] = [float(v) for v in values.values()]
        loss = getattr(obj, "loss", None)
        if callable(loss):
            try:
                out["loss"] = [float(loss())]
            except Exception:
                pass
        elif isinstance(loss, float):
            out["loss"] = [loss]
        return out
''')


def _signatures(objects: dict) -> dict:
    scope: dict = {}
    exec(SIGNATURE, scope)
    return {name: scope["_signature"](obj) for name, obj in objects.items()}


@pytest.mark.parametrize("name", sorted(demos()))
def test_the_generated_script_reproduces_the_session_headless(name):
    session = demo(name)
    expected = {k: v for k, v in _signatures(session.objects).items() if v}
    used = "statsmodels" if statespace._use_statsmodels() else "native"  # the filter the session used, not the setting
    engine = f"import pymetrix.statespace\npymetrix.statespace.ENGINE = {used!r}\n"
    probe = engine + session.script() + SIGNATURE + textwrap.dedent(f'''
        import json, sys
        names = {sorted(expected)!r}
        print("RESULT:" + json.dumps({{n: _signature(globals()[n]) for n in names}}))
        print("LOADED:" + json.dumps(sorted(m for m in sys.modules if m.split(".")[0] in
                                            ("PySide6", "shiboken6", "qtpy", "qtconsole", "matplotlib", "IPython"))))
    ''')
    env = {k: v for k, v in os.environ.items() if k not in ("QT_QPA_PLATFORM", "DISPLAY", "MPLBACKEND")}
    run = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env, timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]
    lines = {line.split(":", 1)[0]: line.split(":", 1)[1] for line in run.stdout.splitlines() if line.startswith(("RESULT:", "LOADED:"))}
    assert json.loads(lines["LOADED"]) == []  # no Qt, no plotting, no IPython
    got = json.loads(lines["RESULT"])
    for obj, parts in expected.items():
        for part, values in parts.items():
            assert np.allclose(got[obj][part], values, rtol=1e-8, atol=1e-10, equal_nan=True), (name, obj, part)
