---
summary: open a Python script in a Code window, run it or a part of it, and carry on from its results with the dialogs.
audience: people who work in code, and EViews users who run programs
order: 10
---

# Run a script in the app

Open a Python script in the app, run it, and find its results in the workspace beside those of the dialogs: each
statement it runs is a step in History, and its objects can be opened, tested and reported like any other. The
script stays an ordinary Python file, which runs the same way outside the app.

## Before you start

- A script. This page uses `consumption.py`, which estimates the error-correction equation for consumption of
  [Estimate an equation and test its residuals](estimate-and-test-an-equation.md) and works out its speed of
  adjustment:

  ```text consumption.py
  from pymetrix import Dataset, Equation

  data = Dataset.example("sample_macro")
  ecm = Equation("dlog(c) ~ dlog(y) + log(c)[-1] + log(y)[-1]").fit(data)
  speed = -float(ecm.params["log(c[-1])"])
  print(f"speed of adjustment: {speed:.2f}")
  ```

- The Immediate window, a Jupyter console, for the script's printed output: it comes with `pip install pymetrix[qt]`
  (qtconsole). Without it scripts still run, and an error is shown in a message box instead.
- An EViews program is not a Python script: [Translate an EViews program](translate-an-eviews-program.md) turns one
  into a script first.

## In the app

1. File → Open script…, and choose `consumption.py`. It opens in a Code window, with its name in the title.
2. ▶ Run (F5) runs the whole script through the Immediate window, one statement at a time. `data`, `ecm` and `speed`
   appear in Objects, the printed line in the Immediate window ("speed of adjustment: 0.75"), and each statement in
   History, as a step of its own.
3. Edit a line, then Run selection (F9) runs the selected lines, or the line the cursor is on, and moves to the next
   line. A line that fails is highlighted, the run stops there, and History shows the step in red.
4. Open `ecm` from Objects and carry on with the dialogs: Next ▸ → Test…, say, or Test → Run test… with Object to
   test: ecm.
5. View → Script shows the session as one script: the code as run, and the dialogs' steps after it as code.
   File → New script from session opens it in a Code window; File → Export script… saves it.
6. Save (Ctrl+S while the Code window has the focus) writes the script back to `consumption.py`.

## In code

```python
import runpy

found = runpy.run_path("consumption.py")      # as `python consumption.py` runs it
found["speed"]                                # 0.75
```

The same script in a session, as the Code window runs it, so that its results are the session's:

```python
from pymetrix.commands import Session

session = Session()
with open("consumption.py", encoding="utf-8") as script:
    session.run_code(script.read(), "editor")
sorted(session.objects)                       # ['data', 'ecm', 'speed']
print(session.script())                       # the script, as View → Script shows it
```

## Check the result

- Objects lists `data`, `ecm` and `speed`; `speed` is 0.75, as when the script runs on its own.
- History has the script's steps, each with its code; a step that failed is red.
- View → Script starts with `from pymetrix import *` and has the script's lines: run in a fresh Python, it gives the
  same objects.

## Pitfalls

- **Files are read from the app's working folder, not the script's.** A script that reads `data.csv` by a relative
  path finds it only if the app was started in that folder. Use full paths, or `Path(__file__).parent` in a script
  run with `python`.
- **Typed code runs only in a trusted project.** A project with code typed or run in the app asks before running it
  when someone else opens it ([Save, reopen and share a project, and replay its script](save-and-share-a-project.md)).
- **A script's names replace the workspace's.** A script that assigns `data` replaces the dataset `data`, and the
  results made from the old one are marked out of date in Objects. Give a script's objects names of their own.
- **Run takes the text in the window, saved or not.** Save it (Ctrl+S) to keep what ran.
- **IPython-only syntax.** `%timeit` and `x?` work in the Immediate window and are recorded, but appear in the script
  as comments.

## See also

- [Reading and writing projects from programs](pymetrix:topic/programs-and-projects), in the help.
- [The user guide's History, Immediate and Code windows](../../../docs/guide.md#history-immediate-and-code-windows):
  completion, Help on a name, and what History can do with the steps.
- [Save, reopen and share a project, and replay its script](save-and-share-a-project.md).
