"""The plugins' public surface and the plugin template: every name pymetrix.extension promises resolves; a plugin made
by ``new_plugin`` imports, registers its call, dialog, views, menu and help, and passes its own tests (an independent
check, the dialog, its tutorial, and the public-surface guard), run in a fresh process as a plugin author would."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from pymetrix.errors import SpecificationError

ROOT = Path(__file__).resolve().parents[1]


def test_every_promised_name_resolves():
    import pymetrix.extension as ext

    for name in ext.__all__:
        assert getattr(ext, name) is not None, name
    from pymetrix._api import api
    from pymetrix.views import _frame_table

    assert ext.api is api and ext.frame_table is _frame_table  # the same objects, under public names
    for module in ext.STABLE:
        __import__(module)
    with pytest.raises(AttributeError):
        ext.nothing_here


def test_a_new_plugin_works_and_passes_its_own_tests(tmp_path):
    from pymetrix.plugin_template import new_plugin

    folder = new_plugin(tmp_path / "pymetrix-demo-stats", "Demo statistics", "Tester")
    names = sorted(str(p.relative_to(folder)).replace("\\", "/") for p in folder.rglob("*") if p.is_file())
    assert names == ["LICENSE", "README.md", "VALIDATION.md", "docs/tutorial.md", "pyproject.toml", "src/pymetrix_demo_stats/__init__.py",
                     "src/pymetrix_demo_stats/_version.py", "src/pymetrix_demo_stats/app.py", "src/pymetrix_demo_stats/core.py",
                     "tests/conftest.py", "tests/test_demo_stats.py", "tests/test_public_surface.py", "tests/test_tutorial.py"]
    assert 'pymetrix.commands"]\ndemo_stats = "pymetrix_demo_stats:register"' in (folder / "pyproject.toml").read_text(encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(folder / "src")]), QT_QPA_PLATFORM="offscreen")
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(folder),
                          "--basetemp", str(tmp_path / "bt"), str(folder / "tests")], cwd=folder, env=env,
                         capture_output=True, text=True, timeout=600)
    assert run.returncode == 0 and "4 passed" in run.stdout, run.stdout[-3000:] + run.stderr[-2000:]
    probe = subprocess.run([sys.executable, "-c", "import pymetrix_demo_stats as p; from pymetrix.commands.registry import COMMANDS; "
                            "from pymetrix import help as h; print(p.PLUGIN.name, COMMANDS['Describe'].cls.command_id, "
                            "'Describing a series' in h.page_for_link(h.link('topic', 'demo_stats-describe')))"],
                           env=env, capture_output=True, text=True, timeout=300)
    assert probe.stdout.split() == ["pymetrix-demo-stats", "demo_stats:describe", "True"], probe.stderr[-2000:]


def test_a_new_plugin_ships_a_licence(tmp_path):
    """The generated plugin carries the BSD-3-Clause text (pymetrix's own LICENSE, with the author and this year as the
    copyright holder) and names it in ``license-files``, so its sdist and wheel include it; its PluginInfo says the
    same licence."""
    import datetime
    import tomllib

    from pymetrix.plugin_template import new_plugin

    folder = new_plugin(tmp_path / "pymetrix-lic", "Licensed", "A. N. Author")
    project = tomllib.loads((folder / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["license"] == "BSD-3-Clause" and project["license-files"] == ["LICENSE"]
    ours = (ROOT / "LICENSE").read_text(encoding="utf-8").splitlines()
    theirs = (folder / "LICENSE").read_text(encoding="utf-8").splitlines()
    assert theirs[0] == "BSD 3-Clause License" and len(theirs) == len(ours)
    assert theirs[2] == f"Copyright (c) {datetime.date.today().year}, A. N. Author"
    assert [line for i, line in enumerate(theirs) if i != 2] == [line for i, line in enumerate(ours) if i != 2]
    assert 'license="BSD-3-Clause"' in (folder / "src" / "pymetrix_lic" / "__init__.py").read_text(encoding="utf-8")


def test_the_names_it_takes():
    from pymetrix.plugin_template import main, new_plugin

    with pytest.raises(SpecificationError, match="pymetrix-myname"):
        new_plugin("my_plugin")
    assert main(["--help"]) == 0
