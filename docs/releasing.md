# Keeping main green and making a release

This page is for maintainers and for anyone (or any AI-assisted session) that pushes to `main`. It was written after
an integration in November 2026 that put eight branches on `main` at once and broke CI three times in a row; each
rule below answers one of the things that went wrong.

## Main is always green

- **Nothing reaches `main` before CI has passed on a branch.** Push the branch (or the WIP branch) and start the
  `tests` workflow on it (Actions → tests → Run workflow, scope `linux`), and wait for every job. `main` then moves by
  fast-forward to that green commit. The push to `main` runs the everyday set again, which also checks `main` itself.
- **Windows on the release commit.** The `release` scope adds Windows and every Python; it runs on the release
  commit (step 4 of the checklist), and on a branch whenever a change needs Windows (fonts, layout, paths, Qt).

## Which jobs run when

The repository is public, so its standard GitHub-hosted runners are free.

| Event | Jobs |
|---|---|
| push to `main`; pull request; Run workflow with scope `linux` (the default); weekly (Mondays) | full on Linux (Python 3.11 and 3.13) and macOS (3.13), oldest supported versions, installed wheel on Linux, no extras (3.13) |
| Run workflow with scope `release` | full on Linux (3.11, 3.12, 3.13), macOS (3.13) and Windows (3.11), installed wheel on Linux, Windows and macOS, no extras (3.11, 3.13), oldest supported versions |

The R comparison for TRAMO-SEATS (`jdemetra-reference.yml`, against rjd3tramoseats) runs weekly and on demand, and on
a pull request only when it touches `packages/pymetrix-seasonal/`, the core modules the plugin builds on
(`seasonal.py`, `jdemetra.py`, `extension.py`) or the workflow itself. Start it on the release commit along with the
`release` scope when the release changes anything it covers.
- **Small batches.** Merge one or two related changes at a time, so that when CI fails it is clear which change
  caused it.
- **One writer at a time.** When several people or sessions are working, only one pushes to `main` at a time; the
  others work through pull requests. If `main` moved while you worked, merge it in and test again before pushing.
- **No fix without evidence.** Reproduce a CI failure, or have a log or stack dump that names its cause, before
  pushing a fix. A plausible explanation that has not been checked is not a fix; say so if a fix is unproven.

## Testing what CI tests

CI's jobs differ from a developer's machine in ways that have each broken `main`:

| CI job | What is different | How to check locally |
|---|---|---|
| full, Windows | the offscreen Qt platform has no fonts: every character is drawn as a box as wide as the font size | run layout and font tests with the box-font emulation used in the November 2026 fixes (commit ee96dd9 describes it) |
| full, Python 3.11 | the newest PySide6 on the oldest Python | keep a 3.11 environment with current dependencies |
| oldest supported versions | every dependency at its lowest pin (`.github/oldest-supported.txt`) | an environment built from those pins |
| installed wheel | the core wheel only: no first-party plugins are registered | run the tests with plugin loading switched off |
| no extras | no optional dependencies | an environment with `pip install .` only |

Tests must not depend on the fonts, screen size, installed plugins or the order tests run in: take expected sizes
from the widgets' own content and font metrics, count only what is installed, and start window tests from a clean
saved layout (the `first_run` fixture in `tests/conftest.py`).

A test worker that crashes writes every thread's stack to a file that CI prints when a job fails; read it before
guessing at a cause. Where a crash report appears in a failed job:

- **Crash reports** (every job): each crashed worker's last tests and every thread's Python stack (faulthandler, one
  `pymetrix-faults-*` file per worker), Java's `hs_err_pid*.log` if the fault was in Java, and on Linux the kernel's
  line naming the faulting library or an out-of-memory kill.
- **Native stacks (Windows)** (the full and installed-wheel jobs on Windows): Windows Error Reporting writes a minidump
  of a `python.exe` that dies of a native fault, and this step prints the faulting thread's C/C++ stack with `cdb`,
  then the top of every thread's stack.
- **Artifacts** of the run (`crash-dumps-full-…`, `crash-dumps-wheel-…`): the dumps themselves, kept five days; open
  one in WinDbg (`File → Open dump file`, then `.ecxr` and `k`) with the same PySide6 version to look further.

## Release checklist

1. **Freeze.** About a day before tagging, stop merging features; only fixes go to `main`.
2. **Green.** CI (scope `linux`) on the commit to be released is green on every job.
3. **Version and notes.** Bump `src/pymetrix/_version.py` (the single source; `pyproject.toml` reads it), move the
   CHANGELOG's "Unreleased" entries under a heading for the new version with the date, and regenerate the API
   reference (`python scripts/make_api_reference.py`) and the defaults snapshot (`python tests/test_defaults_guard.py`).
4. **Green again, with Windows.** Push that commit and run the `tests` workflow on it with scope `release`; wait for
   every job, Windows included. This is the release's one full run. Fast-forward `main` to it.
5. **Tag** that exact commit from a fresh clone, naming the full hash:

   ```bash
   git clone https://github.com/GeoffHeenan/pymetrix.git pymetrix-tag
   cd pymetrix-tag
   git tag -a v0.1.0aN <full-commit-hash> -m "pymetrix 0.1.0aN"
   git push origin v0.1.0aN
   ```

   Type or paste the lines one at a time (Windows PowerShell 5 has no `&&`); some terminals add stray characters around
   a pasted block.
6. **Publish** the GitHub release from the tag, with the CHANGELOG section as its notes.
7. **To PyPI** (when the package is to be installable with `pip`): Actions → release → Run workflow, on the tag, with
   package `pymetrix`. It publishes to TestPyPI, installs from there and runs a demo, then waits for a reviewer before
   PyPI.
8. **Unfreeze.** Open a new "Unreleased" section in the CHANGELOG.

## Releasing a plugin

Each plugin in `packages/` (pymetrix-assistant, -eviews, -finance, -guide, -labbook, -psets, -seasonal) is its own package on
PyPI, released when it changes, by the same `release` workflow:

1. Set its version in `packages/<plugin>/src/<module>/_version.py`. If it needs a newer pymetrix, raise the floor in
   its `pyproject.toml`; that pymetrix must be on PyPI first.
2. Commit, wait for CI, and tag the commit `<plugin>-v<version>` (`pymetrix-seasonal-v0.1.0`, say), as in step 5.
3. Actions → release → Run workflow, on that tag, with the plugin chosen as the package. It checks that the tag and
   the version agree, builds the plugin, publishes it to TestPyPI, installs it from there with pymetrix from PyPI and
   loads each of its entry points, then waits for a reviewer before PyPI.

Each package needs its own trusted publisher, once, on PyPI and on TestPyPI (Your account → Publishing → Add a
pending publisher: the package's name, owner GeoffHeenan, repository pymetrix, workflow `release.yml`, environment
`pypi` or `testpypi`).
