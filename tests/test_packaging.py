"""Packaging: the oldest-supported pins CI tests are exactly the floors pyproject.toml declares; the version is the
same everywhere it is stated; the PyPI page stands on its own."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _floors() -> dict[str, str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    reqs = list(project["dependencies"]) + list(project["optional-dependencies"]["all"])
    out = {}
    for req in reqs:
        # a floor, then any releases left out (a broken one) and where the requirement applies (an environment marker)
        m = re.fullmatch(r"\s*([A-Za-z0-9_.-]+)\s*>=\s*([0-9][^,;\s]*)\s*(,\s*!=\s*[0-9][^,;\s]*\s*)*(;.*)?", req)
        assert m, f"give every dependency a floor (name>=version): {req!r}"
        name = m.group(1).lower()
        assert out.get(name, m.group(2)) == m.group(2), f"{name}: one floor for every environment, not {out[name]} and {m.group(2)}"
        out[name] = m.group(2)
    return out


def _pins() -> dict[str, str]:
    lines = (ROOT / ".github" / "oldest-supported.txt").read_text(encoding="utf-8").splitlines()
    pairs = (line.split("==") for line in lines if line.strip() and not line.startswith("#"))
    return {name.strip().lower(): version.strip() for name, version in pairs}


def _same(a: str, b: str) -> bool:
    pad = lambda v: tuple(int(x) for x in v.split(".")) + (0,) * (3 - len(v.split(".")))  # noqa: E731
    return pad(a) == pad(b)


def test_oldest_supported_pins_are_the_floors():
    floors, pins = _floors(), _pins()
    assert set(floors) == set(pins), f"in one file only: {sorted(set(floors) ^ set(pins))}"
    differ = {name: (floors[name], pins[name]) for name in floors if not _same(floors[name], pins[name])}
    assert not differ, f"floor in pyproject.toml vs pin in .github/oldest-supported.txt: {differ}"


def test_the_version_is_the_same_everywhere():
    """It is written once, in src/pymetrix/_version.py: pyproject.toml reads it from there (hatch's dynamic version),
    and the release workflow checks the tag against it."""
    import pymetrix

    toml = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "version" not in toml["project"] and "version" in toml["project"]["dynamic"]
    assert toml["tool"]["hatch"]["version"]["path"] == "src/pymetrix/_version.py"
    written = re.search(r'^__version__ = "(.+)"$', (ROOT / "src/pymetrix/_version.py").read_text(encoding="utf-8"), re.M).group(1)
    assert pymetrix.__version__ == written
    assert "src/pymetrix/_version.py" in (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    first = re.search(r"^## (.+)$", changelog, re.M).group(1)  # the newest section: "Unreleased" or this version
    assert first == "Unreleased" or first.startswith(f"{written}:"), first


REPO = "https://github.com/GeoffHeenan/pymetrix"


def test_the_pypi_page_links_nowhere_it_cannot_reach():
    """PYPI.md is the package's page on PyPI, where relative links break: any link is a full URL."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    page = (ROOT / project["readme"]).read_text(encoding="utf-8")
    relative = [link for link in re.findall(r"\]\(([^)]+)\)", page) if not link.startswith("https://")]
    assert not relative, f"relative links on the PyPI page: {relative}"


def test_project_urls_point_at_the_public_repository():
    """The core and every plugin name the public repository (PyPI's sidebar): Homepage, Source, Issues and
    Changelog; a plugin's Source is its folder in packages/."""
    tomls = [ROOT / "pyproject.toml"] + sorted((ROOT / "packages").glob("*/pyproject.toml"))
    assert len(tomls) > 1
    for toml in tomls:
        urls = tomllib.loads(toml.read_text(encoding="utf-8"))["project"].get("urls", {})
        source = REPO if toml.parent == ROOT else f"{REPO}/tree/main/packages/{toml.parent.name}"
        assert urls == {"Homepage": REPO, "Source": source, "Issues": f"{REPO}/issues",
                        "Changelog": f"{REPO}/blob/main/CHANGELOG.md"}, toml.relative_to(ROOT)


def test_current_docs_do_not_call_the_project_private():
    """The repository is public (BSD-3-Clause). Historical CHANGELOG headings keep what each build was called; the
    current pages, templates and the Unreleased section do not say "private review" or "not public"."""
    pattern = re.compile(r"private (review|demo|build|repository)|\(private\)|not (yet )?public|until (the repository is )?public|repository is private", re.I)
    pages = ["README.md", "PYPI.md", "CONTRIBUTING.md", "GOVERNANCE.md", "SECURITY.md", "docs/known-gaps.md",
             "docs/roadmap.md", "docs/how-to-help.md", "docs/install.md", "docs/pitch.md",
             ".github/workflows/release.yml", "pyproject.toml", *sorted(str(p.relative_to(ROOT)) for p in (ROOT / ".github" / "ISSUE_TEMPLATE").iterdir())]
    found = [(page, m.group(0)) for page in pages for m in pattern.finditer((ROOT / page).read_text(encoding="utf-8"))]
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    unreleased = changelog.split("\n## ", 2)[1] if changelog.startswith("# Changelog\n\n## Unreleased") else ""
    found += [("CHANGELOG.md (Unreleased)", m.group(0)) for m in pattern.finditer(unreleased)]
    assert not found, found


def test_every_plugin_ships_the_licence():
    """Each first-party plugin in packages/ carries a copy of the root BSD-3-Clause LICENSE and names it in
    ``license-files``, so its sdist and wheel include it (checked by building one: ``LICENSE`` in the sdist and in
    ``*.dist-info/licenses/``)."""
    licence = (ROOT / "LICENSE").read_bytes()
    plugins = sorted(p for p in (ROOT / "packages").iterdir() if (p / "pyproject.toml").exists())
    assert plugins
    for plugin in plugins:
        project = tomllib.loads((plugin / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        assert project["license"] == "BSD-3-Clause", plugin.name
        notices = ["THIRD-PARTY-NOTICES.md"] if (plugin / "THIRD-PARTY-NOTICES.md").exists() else []
        assert project.get("license-files") == ["LICENSE", *notices], plugin.name
        assert (plugin / "LICENSE").read_bytes() == licence, plugin.name


def test_plugins_ship_notices_for_third_party_data():
    """A plugin that bundles a dataset from elsewhere (its ``meta.license`` is not "simulated by pymetrix") ships a
    THIRD-PARTY-NOTICES.md naming it, in ``license-files`` so that its sdist and wheel include it (checked by building
    them); for linearmodels' NCSA data, with linearmodels' notice reproduced, as that licence asks for binary
    redistribution. The notice names every bundled dataset, simulated or not."""
    import json

    import pyarrow.parquet as pq

    checked = 0
    for plugin in sorted(p for p in (ROOT / "packages").iterdir() if (p / "pyproject.toml").exists()):
        files = sorted((plugin / "src").rglob("*.parquet"))
        licences = {f.stem: json.loads(pq.read_schema(f).metadata[b"pymetrix"])["meta"]["license"] for f in files}
        outside = {name: lic for name, lic in licences.items() if not lic.startswith("simulated by pymetrix")}
        if not outside:
            continue
        project = tomllib.loads((plugin / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        assert "THIRD-PARTY-NOTICES.md" in project["license-files"], plugin.name
        notice = (plugin / "THIRD-PARTY-NOTICES.md").read_text(encoding="utf-8")
        missing = [name for name in licences if f"datasets/{name}.parquet" not in notice and f"`{name}`" not in notice]
        assert not missing, (plugin.name, missing)
        if any("linearmodels" in lic for lic in outside.values()):
            assert "> Copyright (c) 2017 Kevin Sheppard. All rights reserved." in notice, plugin.name
        checked += len(outside)
    assert checked >= 1  # the finance plugin's treasury_yields (its french data are downloaded on first use, not shipped)
