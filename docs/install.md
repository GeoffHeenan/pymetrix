# Installing pymetrix

pymetrix needs Python 3.11 or later (3.13 is the main test platform) on Windows, macOS or Linux. It
is not on PyPI yet: the development builds are attached to releases of the GitHub repository.
Install into a virtual environment of its own, so nothing else on your machine changes.

## System requirements

| | Minimum | Comfortable |
|---|---|---|
| Python | 3.11, 64-bit | 3.13 |
| Disk | 0.5 GB for the core; 1.5 GB with `[all]` (the desktop app is half of it) | |
| Memory | 4 GB | 8 GB, for parallel chains alongside the app |
| Processor | any 64-bit | 4 or more cores, for multi-chain sampling and estimation from several starts |

Measured with 0.1.0a5 (peak memory):
- Scripts and notebooks: about 150 MB to import pymetrix, and 160 to 180 MB for the demo projects, estimation and
  Metropolis-Hastings on the New Keynesian model included.
- The desktop app: about 370 MB with a project and a few result windows open.
- Each chain or estimation start run in parallel: about 180 MB, even for Smets and Wouters (2007). Linear algebra runs
  on one core per process during estimation, so chains and starts run about as many at once as there are cores.
- Posterior draws: 8 bytes per number, so chains x draws x parameters x 8 bytes. That is about 50 MB for 4 chains of
  40,000 draws of 36 parameters. A very long run (4 x 400,000) needs about 2 GB in all.

## 1. A virtual environment

Windows (PowerShell):

```powershell
py -3.13 -m venv pymetrix-env
pymetrix-env\Scripts\Activate.ps1
```

macOS and Linux:

```bash
python3 -m venv pymetrix-env
source pymetrix-env/bin/activate
```

## 2. pymetrix

**From the release's wheel.** Download `pymetrix-0.1.0a19-py3-none-any.whl` from the release page,
then, in the folder where it is:

```bash
python -m pip install "pymetrix-0.1.0a19-py3-none-any.whl[all]"
```

**Or straight from the repository**, if you have access to it (git must be installed):

```bash
python -m pip install "pymetrix[all] @ git+https://github.com/GeoffHeenan/pymetrix@v0.1.0a19"
```

`[all]` adds the desktop app (PySide6), linearmodels, arch, scikit-learn, Jupyter forms and the report
writers: about 1.3 GB on disk, mostly the app (PySide6; see the requirements above). The core alone (`.whl` without `[all]`) needs NumPy, pandas, SciPy, statsmodels,
matplotlib, pyarrow and threadpoolctl; add `[qt]` for the app, `[jupyter]` for forms in notebooks, and `[mcp]` for the server that lets an AI assistant
use pymetrix (`pymetrix-mcp`; see the user guide).

**X-13ARIMA-SEATS and TRAMO-SEATS** are in the seasonal plugin, `pymetrix-seasonal`, a package of its own (STL and
classical decomposition are in the core). From the repository: `python -m pip install -e "packages/pymetrix-seasonal[jdemetra]"`
(or `"pymetrix-seasonal[jdemetra] @ git+https://github.com/GeoffHeenan/pymetrix#subdirectory=packages/pymetrix-seasonal"`).
X-13 needs the Census Bureau's `x13as` program on the PATH or in `X13PATH`. TRAMO-SEATS runs Eurostat's JDemetra+ in
Java: install Java 21 or later (for example Eclipse Temurin), JPype (the plugin's `[jdemetra]` extra), and JDemetra+'s
libraries, once: `python -c "import pymetrix_seasonal.jdemetra as j; j.install()"` downloads them (about 7 MB, from Maven Central) to `~/.pymetrix/jdemetra`. Offline, copy the jars there or to a folder
named by `JDPLUS_PATH`.

**The finance plugin** (value at risk, multivariate GARCH, copulas, portfolios, options, yield curves, credit) is a
package of its own, `pymetrix-finance`. From the repository: `python -m pip install
"pymetrix-finance @ git+https://github.com/GeoffHeenan/pymetrix#subdirectory=packages/pymetrix-finance"` (or, in a
clone, `python -m pip install -e packages/pymetrix-finance`); once it is on the package index, `pymetrix[finance]`.

**EViews workfiles** come in through the EViews plugin, `pymetrix-eviews`, a package of its own: in a clone,
`python -m pip install -e packages/pymetrix-eviews` (or `"pymetrix-eviews @ git+https://github.com/GeoffHeenan/pymetrix#subdirectory=packages/pymetrix-eviews"`).
Its export program, run in EViews, writes a bundle that File → Import EViews workfile… reads. The program is experimental (it
has not been run in EViews); saving the workfile from EViews as CSV or Excel needs no plugin and is the supported way.

**Or nothing to install:** open the repository in GitHub Codespaces (see the README), which runs the
desktop app in your browser.

## 3. Check it works

```bash
python -c "import pymetrix; print(pymetrix.__version__); print(pymetrix.demo('nk_estimation')['posterior'].summary())"
pymetrix-gui --demo nk_estimation
```

The first prints the version and a posterior from a small New Keynesian model (about a minute the first time). The
second opens the desktop app on the same demo, built one dialog at a time: **History** lists the
steps and **Script** shows them as Python. **File → Open demo project…** lists the others.

## Updating

From a new release's wheel or tag, run the same `pip install` with the new version in it. In a clone installed with
`python -m pip install -e ".[all]"`, `git pull` brings the new code at once, but pip's record of the package (the
version `pip show pymetrix` reports, the dependencies, the `pymetrix-gui` and `pymetrix-mcp` commands) is the one made
at install time: run the same `pip install -e` again after a pull that changes the version or the dependencies.
`pymetrix.__version__` always reads the source (`src/pymetrix/_version.py`, the one place the version is written).

## Where to start

- [Start here: building and estimating models](modelling-quickstart.md), for Dynare and IRIS users.
- The [README](../README.md) quick start and course labs, for data, estimation and forecasting.
- [Validation](validation.md): what the numbers are checked against.
- [Known gaps](known-gaps.md): what is missing or rough. Please say when something blocks you.

## Problems

- **A desktop shortcut:** `pymetrix-gui --shortcut` (with the environment active, once) puts a pymetrix shortcut
  on the desktop and in the Start menu (Windows), the applications menu (Linux) or the desktop (macOS). It starts
  that environment's Python, so it works later without activating anything. If you installed from a clone of the
  repository (`pip install -e .`), `pymetrix-gui --shortcut --latest` makes a shortcut that pulls the latest commits
  of the checked-out branch each time it starts, so the icon always runs the most recent build. A pull that cannot be
  made (offline, or local changes in the way) is reported on the opening screen and the app starts as it is; new
  dependencies still need `pip install -e .` again.
- **`pymetrix-gui` is not found:** the virtual environment is not active; activate it (step 1), or run
  `python -m pymetrix.ui.qt`.
- **"plugin finance could not load: could not be imported: its code can't be found…":** the plugin's metadata is
  installed but its code is not where it was installed from. Most often it was installed from a clone with
  `pip install -e packages/pymetrix-finance` and that folder was then moved, deleted or checked out at a branch
  without it. **Help → Plugins** (choose the plugin) says which package, whether it is an editable install and from
  which folder, and gives the command for the Python the app runs (**Copy command** copies it): reinstall it from your
  clone (`python -m pip install -e <clone>/packages/pymetrix-finance`) or remove it (`python -m pip uninstall
  pymetrix-finance`). "Needs *module*, which isn't installed" instead means the plugin is there but a library it uses
  is not: `python -m pip install` that library.
- **The app does not start on Linux:** PySide6 needs the system's OpenGL and XCB libraries
  (`sudo apt install libegl1 libxkbcommon-x11-0 libxcb-cursor0` on Ubuntu).
- **Estimation or sampling is slow on a laptop:** pymetrix runs them with one linear-algebra thread,
  which is usually fastest for models of this size; `python benchmarks/model_speed.py` (in the
  repository) times each step on your machine.
