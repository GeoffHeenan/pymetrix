---
summary: save the session as one project file, reopen it with every result rebuilt, send it to a colleague, and run its script without the app.
audience: everyone, and people who work in code
order: 6
---

# Save, reopen and share a project, and replay its script

Save the work of a session as a project: one `.pymx` file with the data it started from and every step, dialogs and
typed code alike. Opening it replays the steps and rebuilds each result. The same steps are a Python script, which
runs without the app; a colleague can open the project, or run the script.

## Before you start

- Some work in the session. This page loads the example `sample_macro`, estimates the error-correction equation for
  consumption of [Estimate an equation and test its residuals](estimate-and-test-an-equation.md), and types one line
  of code that uses it.
- Where the files go. A project is one file; data read from files on your disk is kept inside it, so it opens on
  another computer.

## In the app

1. File → Load example data…, with Example dataset: sample_macro and Save as: data. Estimate → Equation…, with
   Formula: `dlog(c) ~ dlog(y) + log(c)[-1] + log(y)[-1]` and Save as: ecm.
2. In View → Immediate, type `speed = -float(ecm.params['log(c[-1])'])` and press Enter: the speed at which
   consumption returns to its long-run ratio to output. Typed code is a step too: History lists it beside the
   dialogs' steps.
3. File → Save project as…, and name the file `consumption.pymx`. File → Save project (Ctrl+S) saves it again later
   under the same name.
4. File → New session, then File → Open project… and choose `consumption.pymx`. The steps run again: `data`, `ecm`
   and `speed` are back in Objects, and History has the steps. File → Open recent lists the projects opened lately.
5. To share it, send the file. When a colleague opens it, the app shows the typed code first and asks: Run the code
   and trust this project, or Open without running the code (the dialogs' steps run either way; the typed code is
   kept, marked not run).
6. File → Export script…, and name it `consumption.py`: the session as a Python script, the same text as View →
   Script. It runs with `python consumption.py`, without the app.

## In code

```python
from pathlib import Path
import subprocess
import sys

from pymetrix.commands import FitEquation, LoadExample, Session, load_project, save_project
from pymetrix.commands.project import project_code

session = Session()
session.run(LoadExample("sample_macro", name="data"))
session.run(FitEquation("dlog(c) ~ dlog(y) + log(c)[-1] + log(y)[-1]", name="ecm"))
session.run_code("speed = -float(ecm.params['log(c[-1])'])")       # 0.75, typed in the Immediate window
save_project(session, "consumption.pymx")
```

Reopen it, as you and as a colleague who has not trusted its code; then run its script:

```python
mine = load_project("consumption.pymx")                 # data, ecm and speed, rebuilt
theirs = load_project("consumption.pymx", trust=False)   # data and ecm: the typed code is kept, not run
project_code("consumption.pymx").steps                   # the typed code, to read before trusting it
Path("consumption.py").write_text(session.script(), encoding="utf-8")
subprocess.run([sys.executable, "consumption.py"], check=True)   # the script, in a fresh Python
```

## Check the result

- After reopening, the results are the same, to the last digit: `ecm`'s coefficients, and `speed`, 0.75. The project
  keeps the steps, not the results, so they are made again; a slow step's result (two seconds or more) is kept as
  well, and taken from the project when you open your own.
- The colleague's copy without trust: `data` and `ecm`, not `speed`, and a warning that the typed code
  was not run.
- `consumption.py` has a line for each step, `ecm = Equation('dlog(c) ~ dlog(y) + log(c)[-1] + log(y)[-1]').fit(data,
  method=OLS())` among them, and runs without error in a Python with pymetrix.

## Pitfalls

- **Data is kept as it was read, until you refresh it.** A project keeps the data of the files and the data services
  it loaded (File → Load data…, File → Load from connector…), so it opens with the same numbers when the file has
  moved or the provider has revised its series. File → Refresh data… reads them again, re-runs the steps that changed
  them and recomputes what depends on them; save the project to keep the new data. The script, run on its own,
  always reads the sources as they are then.
- **The script reads files by their path.** `Dataset.read('quarterly.csv', ...)` in the script needs the file where
  it was; the project does not, as it keeps the data inside. Send the data with the script, or send the project.
- **Code is trusted by its fingerprint.** Change one character of a project's typed code and it asks again. Read the
  code (`project_code(path).steps`) before trusting a project from someone else: Python can do anything a program
  can.
- **A project needs the plugins its steps used.** Opening one whose steps came from a plugin that is not installed
  says which to install.
- **To send one result, not the work,** use File → Save result… (`.pymxr`): its tables and figures as they were,
  which can be read but not estimated again.

## See also

- [Reading and writing projects from programs](pymetrix:topic/programs-and-projects), in the help.
- [Run a script in the app](run-a-script-in-the-app.md): the other way round, a script's steps recorded in the
  session.
- [The user guide's Projects](../../../docs/guide.md#projects) and
  [Reading and writing projects from programs](../../../docs/guide.md#reading-and-writing-projects-from-programs):
  what a project keeps, and how other programs read and write one.
