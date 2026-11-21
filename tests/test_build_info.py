"""The pymetrix commit (and whether the code differed from it) beside the version: in provenance, a project's save_log
and a saved result's manifest; an installed package records the version only and says so. Checked against git run
directly in the test."""

import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

import pymetrix
from pymetrix import build
from pymetrix.build import INSTALLED, build_info

PACKAGE = Path(pymetrix.__file__).resolve().parent
HAS_GIT = shutil.which("git") is not None


def _git(folder, *args):
    return subprocess.run(["git", "-C", str(folder), *args], capture_output=True, text=True, check=False)


def _tracked() -> bool:
    return HAS_GIT and _git(PACKAGE, "ls-files", "--error-unmatch", "__init__.py").returncode == 0


@pytest.mark.skipif(not _tracked(), reason="pymetrix is not running from a git checkout")
def test_the_commit_and_dirty_flag_of_this_checkout():
    info = build_info()
    assert info["commit"] == _git(PACKAGE, "rev-parse", "HEAD").stdout.strip()
    changed = _git(PACKAGE, "status", "--porcelain", "--untracked-files=no", "--", ".").stdout.strip()
    assert info["dirty"] is bool(changed) and "note" not in info


def test_read_once_per_process(monkeypatch):
    first = build_info()

    def no_git(*args, **kwargs):
        raise AssertionError("git run again")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert build_info() == first
    assert build_info() is not build_info()  # a copy each time: changing one changes nothing else


def test_an_installed_package_records_the_version_only(tmp_path):
    folder = tmp_path / "site-packages" / "pymetrix"
    folder.mkdir(parents=True)
    (folder / "__init__.py").write_text("")
    assert dict(build._read.__wrapped__(str(folder))) == {"commit": None, "note": INSTALLED}


@pytest.mark.skipif(not HAS_GIT, reason="git is not installed")
def test_a_package_inside_someone_elses_checkout_is_not_theirs(tmp_path):
    """A virtual environment inside another project's repository: that repository's commit is not pymetrix's."""
    assert _git(tmp_path, "init", "-q").returncode == 0
    folder = tmp_path / ".venv" / "lib" / "pymetrix"
    folder.mkdir(parents=True)
    (folder / "__init__.py").write_text("")
    assert dict(build._read.__wrapped__(str(folder))) == {"commit": None, "note": INSTALLED}


@pytest.mark.parametrize("problem", [FileNotFoundError("no git"), subprocess.TimeoutExpired("git", build.TIMEOUT)])
def test_no_git_or_a_slow_one_gives_no_commit_and_never_raises(tmp_path, monkeypatch, problem):
    (tmp_path / ".git").mkdir()

    def fails(*args, **kwargs):
        assert kwargs["timeout"] == build.TIMEOUT  # always with a time limit
        raise problem

    monkeypatch.setattr(subprocess, "run", fails)
    assert dict(build._read.__wrapped__(str(tmp_path)))["commit"] is None


def test_in_provenance_and_shown(tmp_path):
    from pymetrix import Dataset, Equation
    from pymetrix.views import provenance_table, _commit_text

    fit = Equation("c ~ y").fit(Dataset.example("sample_macro"))
    assert dict(fit.provenance.build) == build_info()
    table = provenance_table(fit.provenance)
    shown = {r[0]: r[1] for r in table.rows}
    assert shown["commit"] == _commit_text(build_info())
    assert _commit_text({"commit": "0123456789abcdef" * 2 + "01234567", "dirty": True}) == "0123456789ab, with uncommitted changes"
    assert _commit_text({"commit": None, "note": INSTALLED}) == INSTALLED and _commit_text(None) is None


def test_old_saved_provenance_does_not_take_todays_commit():
    from pymetrix.codec import decode, encode
    from pymetrix.metadata import Provenance

    prov = Provenance({}, {}, {}, None, 10, 0, package_version="0.1.0a1")
    value = encode(prov)
    assert value["build"] == build_info()
    del value["build"]  # as saved before the commit was recorded
    old = decode(value)
    assert old.build["commit"] is None and "not recorded" in old.build["note"]
    assert Provenance({}, {}, {}, None, 10, 0).build is None  # no version given: nothing claimed


def test_in_the_save_log_and_a_saved_results_manifest(tmp_path):
    from pymetrix.commands import FitEquation, LoadExample, Session, load_project, save_project
    from pymetrix.saved import load_result, save_result
    from pymetrix.views import view_sections

    s = Session()
    s.run(LoadExample())
    s.run(FitEquation("c ~ y", name="fit"))
    path = save_project(s, tmp_path / "p.pymx")
    with zipfile.ZipFile(path) as z:
        entry = json.loads(z.read("manifest.json"))["save_log"][-1]
    assert entry["pymetrix"] == pymetrix.__version__ and {k: entry[k] for k in build_info()} == build_info()
    assert load_project(path).save_log[-1]["commit"] == build_info()["commit"]

    saved_path = save_result(s["fit"], tmp_path / "fit")
    with zipfile.ZipFile(saved_path) as z:
        manifest = json.loads(z.read("manifest.json"))
    assert {k: manifest[k] for k in build_info()} == build_info()
    saved = load_result(saved_path)
    assert dict(saved.build) == build_info()
    tab = dict(view_sections(saved))["Saved"]
    assert ("PyMetrix commit" in [r[0] for r in tab.rows]) == bool(build_info()["commit"] or build_info().get("note"))
