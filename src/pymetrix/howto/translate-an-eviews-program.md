---
summary: translate an EViews program (.prg) into a pymetrix script with a report on every line, bring its workfile's data across, and run it as steps.
audience: EViews users with programs
order: 15
plugin: pymetrix-eviews
---

# Translate an EViews program

Turn an EViews program into a readable pymetrix script, with a report that gives every line of the program a
status (translated, translated with a note, left out, error) and the reason; then run the program's work as steps of
the session, each recorded and undoable. The data the program opens come across as a CSV or Excel file saved from
EViews. All of this is the `pymetrix-eviews` plugin's, a separate install.

## Before you start

- The plugin, `pymetrix-eviews`: `pip install --pre pymetrix-eviews` (or, in a clone of pymetrix,
  `pip install -e packages/pymetrix-eviews`). Restart the app and, when it asks about the new plugin, choose Enable: the `pymetrix-eviews`
  plugin adds File → Translate EViews program…, File → Import EViews workfile… and the View → EViews translation
  panel.
- The workfile's data, as the program's `wfopen` expects them. The supported way is to save the page from EViews as
  CSV or Excel (Save As in EViews's File menu, or `wfsave(type=csv) macro.csv`). This page uses six years of
  consumption and income, as EViews saves a quarterly page, its dates in the `obs` column, as `macro.csv`:

  ```text macro.csv
  obs,cons,inc
  2018Q1,373.8,482.0
  2018Q2,373.5,482.5
  2018Q3,377.2,487.1
  2018Q4,380.3,491.4
  2019Q1,386.8,499.8
  2019Q2,390.7,504.6
  2019Q3,395.7,512.9
  2019Q4,398.7,513.2
  2020Q1,397.2,512.3
  2020Q2,395.9,513.5
  2020Q3,394.4,511.3
  2020Q4,401.2,519.8
  2021Q1,406.4,526.5
  2021Q2,402.1,521.9
  2021Q3,407.4,529.1
  2021Q4,411.1,531.6
  2022Q1,414.8,536.4
  2022Q2,417.2,539.6
  2022Q3,418.0,541.2
  2022Q4,421.1,544.4
  2023Q1,425.7,551.5
  2023Q2,427.0,553.3
  2023Q3,422.9,549.2
  2023Q4,421.4,547.1
  ```

- The program, `consumption.prg`: growth rates, a least-squares equation with HAC standard errors, a scalar from
  it, a serial-correlation test and a graph:

  ```text consumption.prg
  ' Consumption and income, saved from the workfile as macro.csv
  wfopen macro.csv
  series dlc = 100*dlog(cons)
  series dly = 100*dlog(inc)
  smpl 2018q2 @last
  equation eq_c.ls(cov=hac) dlc c dly
  scalar mpc = eq_c.@coefs(2)
  eq_c.auto(2)
  graph gr.line dlc dly
  ```

- The other way in, a bundle, is experimental. The plugin's export program, run in your EViews, writes the open
  workfile as a bundle: each page's series with their labels, units and sources, and its groups, samples, scalars,
  matrices, equations (with EViews's estimates) and models. It was written from EViews's documentation and has not
  been run in EViews; until it has, use CSV or Excel.

## In the app

1. File → Translate EViews program… (the `pymetrix-eviews` plugin's), with Program: consumption.prg, Workfile data: macro.csv, Date column: obs and
   Frequency: Quarterly. Program arguments are what follows the program's name in EViews's `exec` (`%0`, `%1`);
   none here. The step translates only: it changes no data.
2. The EViews translation panel opens (View → EViews translation, docked at the bottom): a row a line, with its
   number, its status in words and in colour, the EViews line, the Python it became and the reason. The filters
   give each status with its count: translated (5), translated with a note (3), left out (1), error (0).
3. Select a row: the program and the script beside the table both scroll to that line. Line 6, the equation, is
   translated with a note: EViews's HAC bandwidth and kernel may differ from pymetrix's, so check them. Line 9, the
   graph, is left out, with pymetrix's way to draw one.
4. Copy script copies the whole script (Copy, the rows selected or the whole report). In a Code window it runs as it
   is (File → New script, paste, ▶ Run).
5. Run as steps runs the program's work in this session, one step at a time: the dataset `macro` (read from
   `macro.csv`, with `dlc` and `dly`), the equation `eq_c` and its test `eq_c_auto` appear in Objects, each a step in
   History that Edit → Undo takes back. The run stops at the first step that fails and marks its row "failed".
6. For a bundle instead: File → Import EViews workfile…, with Import: Get the export program (experimental) and
   Save the program as a path, saves `pymetrix_export.prg`. Run it in EViews on the open workfile
   (`exec pymetrix_export.prg C:\work\macro-export`), then give `macro.evbundle` as Workfile data, or read its pages
   with File → Import EViews workfile…, Import: Workfile bundle and Bundle: macro.evbundle.

## In code

```python
from pymetrix.commands import Session
from pymetrix_eviews import save_export_program, translate_program

t = translate_program("consumption.prg", bundle="macro.csv", read_options={"time": "obs", "freq": "Q"})
print(t.script)                         # each EViews line as a comment, above what it became
t.report.counts()                       # translated 5, with a note 3, left out 1, error 0
t.report.summary()                      # line, status, EViews, Python, reason
session = Session()
made = t.run(session)                   # Run as steps: macro, eq_c and eq_c_auto in the session
made["mpc"]                             # 1.00: the scalar, eq_c's coefficient on dly
save_export_program(".")                # pymetrix_export.prg, for a bundle (experimental)
```

## Check the result

- The report: nine rows, one for each line; nothing is dropped silently. The counts in the panel's filters are
  those of `t.report.counts()`.
- The script runs on its own and gives what the steps give: `eq_c`'s coefficients, -0.03 and 1.00, and the
  Breusch-Godfrey test with 2 lags.
- The equation against EViews's own output for the same program: the coefficients must agree to the last digit; the
  HAC standard errors only if the bandwidth and kernel are the same (the note on line 6 says so).
- `mpc` is `eq_c`'s coefficient on `dly`: EViews's `@coefs(2)`, the second in the list `c dly`.
- The equation's sample: 2018Q2 to 2023Q4, from `smpl 2018q2 @last`, where the growth rates start: 23 quarters.

## Pitfalls

- **The plugin is a separate install.** Without `pymetrix-eviews`, File → Translate EViews program… is not in the menu, and a
  project or script that uses `translate_program` or a bundle asks for it. Steps run from the translation are
  ordinary pymetrix steps, which do not.
- **The workfile's binary file (.wf1) is not read.** Its format is not published. Save the page as CSV or Excel
  from EViews, and give its date column and frequency.
- **A CSV file carries the numbers only.** Labels, units, groups, samples and EViews's estimates do not come with
  it; a bundle carries them, but its export program is experimental.
- **Some lines are left out, each with its reason.** Graphs, tables and spools, databases, add-ins, EViews's date
  numbers, and estimators pymetrix does not have (`stepls`, `breakls`, `threshold`, `switchreg`): the reason names
  pymetrix's way where there is one. An option that is not mapped leaves the line out, so an estimate never runs
  without it and differs silently.
- **Notes say what to check.** Where EViews's defaults differ from pymetrix's (the HAC bandwidth, ARMA and GARCH
  estimators, the model solver), the line is translated with a note: compare those numbers with EViews's before
  relying on them.
- **The data must be what the program expects.** An `error` row names something the program has not made, often a
  series missing from the CSV file, or one named differently.

## See also

- The plugin's help topics, EViews programs and EViews workfiles, in Help → Help contents once it is installed.
- [Bring an EViews model across, then check and solve it](bring-in-an-eviews-model.md): a model's text alone, in
  the model editor, without the plugin.
- [Run a script in the app](run-a-script-in-the-app.md): the translated script, run and changed.
- [The plugin's README](../../../packages/pymetrix-eviews/README.md): every EViews command translated, by phase, and
  the bundle's format; [the user guide's EViews programs](../../../docs/guide.md#eviews-programs);
  [Coming from EViews](pymetrix:topic/coming-from-eviews), in the help.
