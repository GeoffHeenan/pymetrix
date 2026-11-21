# Writing a case study

For lecturers and trainers. A case study takes a trainee through a workflow in the desktop app, one step at a
time, through the app's real dialogs, and ticks each step as its result appears. It needs the `pymetrix-guide`
plugin (`pip install pymetrix[guide]`; experimental), which adds the **Guide** panel (View → Guide).

A case study is an ordinary Markdown note in a course folder, so it goes wherever your course material goes:
**Help → Course material…**, the `PYMETRIX_COURSES` variable for a lab, a package with a `pymetrix.courses` entry
point, or a project (`session.attach_course(folder)`), which saves the folder with the project. The plugin's own
example is Lab 3, the forecast round, in `packages/pymetrix-guide/src/pymetrix_guide/courses/case_studies`.

## 1. Start from a lab you have

Take a lab whose steps are menu commands. Write its text as you would for the lab, and after each instruction put
a `pymetrix-step` block:

````markdown
---
title: Week 1, guided. The consumption function
course: ECON 101
links: [commands/FitEquation]
---

# The consumption function

Estimate the static consumption function first: consumption on income (**Estimate → Equation**).

```pymetrix-step
title: The static consumption function
command: FitEquation
settings: {formula: "c ~ y", name: static}
check: object static is FittedEquation
```

Is the residual autocorrelated? Add lagged consumption.

```pymetrix-step
title: The dynamic consumption function
command: FitEquation
settings: {formula: "c ~ y + c[-1]", name: dynamic}
check: object dynamic is FittedEquation
```

Compare the two marginal propensities to consume: which is the long-run one?
````

The text before a block is that step's text; the text after the last block is shown when the trainee finishes.
A short case study like this one works as it is. A longer one needs the context that section 2 adds.

## 2. Give the context: an introduction, stages, and why each step

A trainee who steps through dialogs without knowing where the steps lead learns the menus, not the method. Give
them three things.

**An introduction.** Put `##` sections before the first step. Each is a page of its own, shown before step 1, and
**Back** and **Next** move through them. Write two at least:

- *the workflow in general*: its stages, why each is done, and how the app supports it, in the setting of the
  example (for a forecast round: the data and their vintage, the model, the baseline, judgement, scenarios,
  evaluation, the report);
- *this scenario*: the data, the period, the model and why it was chosen, the judgement or assumption it makes and
  why, what is compared with what, and what the trainee should be able to explain at the end.

Keep each page to about a screen. The first step's own text follows a line `---` (after a blank line) below the
last section. Without `##` sections, everything before the first block is the first step's text, as in section 1.

**Stages.** Name the stage of the workflow each step belongs to with `stage:` in its block; a step without one is
in the same stage as the step before. List the stages in order in the front matter (`stages: [...]`), or they are
taken from the steps. The panel shows them as a strip with the current one in bold, and the progress as
"Stage: Judgement — step 8 of 19".

**Why, and what to look for.** Give each step `why:`, its purpose in a sentence, and `look:`, what to look at in
its result: "the residual autocorrelations are small", "the judgement moves the path by much less than the width
of the fan". The step's text says what to do; these say why it matters. Each is one line.

A `summary:` in the front matter is shown under the title when the case study is chosen. The text after the last
step is the finished page: recap what was done and why, and say where to go next (the lab notebook, the course
pack's chapters, the related dialogs).

````markdown
---
title: Week 1, guided. The consumption function
summary: A static and a dynamic consumption function, and which marginal propensity is the long-run one
stages: [Data, Model, Interpret]
course: ECON 101
links: [commands/FitEquation]
---

# The consumption function

## Overview: estimating a behavioural equation

Applied work goes from the data, to a model, to checking it, to reading its coefficients...

## This scenario

Consumption and income in a simulated economy, 2000Q1 to 2024Q4. The question: how much of an extra unit of
income is spent, at once and in the long run? ...

---

Estimate the static consumption function first: consumption on income (**Estimate → Equation**).

```pymetrix-step
title: The static consumption function
stage: Model
command: FitEquation
settings: {formula: "c ~ y", name: static}
check: object static is FittedEquation
why: The simplest version: a benchmark for the dynamic one.
look: The Durbin-Watson statistic: well below 2, the residuals are positively autocorrelated, a sign of missing dynamics.
```
````

The plugin's Lab 3 is a full example: four introduction pages, seven stages, and a why and a look for each of its
19 steps.

## 3. Name the dialog and its settings

`command` is the dialog's class name. The help page of every dialog gives it (Help → the dialog, or the **Help**
button in the dialog), and `pymetrix.commands.COMMANDS` lists them all. `settings` are its fields, by the names
the help page gives, on one line as `{field: value, ...}` or as indented lines:

```text
settings:
  variables: "ly, lc, li"
  lags: 2
  name: var
```

Quote text that holds a comma, colon or brace. Fields you leave out keep the dialog's defaults. **Open…** fills
the dialog with these values and lets the trainee change them; **Show me** runs them as written.

## 4. Say what ticks the step

`check` says when the step is done. It is a fixed set of conditions, not code, so a case study can be shared
freely:

- `object fit exists`
- `object fit is FittedEquation` (the result's class, or one it derives from; in code, `type(session["fit"]).__name__`)
- `series ly in data`
- `step AdjustRound ran`, `step AdjustRound ran on forecast_round` (for steps that change an object in place)
- several joined with `and`

Check the result, not the route: if the trainee changes the settings, the step still ticks when its result is
there. Undo takes the tick back, and Redo gives it again.

## 5. Choose where it starts

With `demo: consumption` in the front matter, the case study starts from that demo project
(`pymetrix.demos()` lists them). The panel offers to open it, and **Start over** opens it again as it first was.
Either asks to save unsaved work first, as File → Open demo project does. (With a pymetrix before extension API
1.7 the panel cannot open a demo itself, and says where to find it in the File menu.)
Choose a demo that does not already hold the results your steps make, or their ticks will show at once.
Without a demo the case study starts from whatever is open; say in the text if it needs an empty project
(**File → New session**). `level: introductory` is shown next to the title in the panel's list.

## 6. Check it before handing it out

```python
from pymetrix_guide import check_case_study
from pymetrix.testing import check_course

check_case_study("econ101/week1_guided.md")   # each problem with its line: a dialog or field that does not exist, a bad check
check_course("econ101")                        # the course folder: links and files
```

`check_case_study` raises on problems, and warns (`CaseStudyWarning`) about what would make the case study
better: no introduction, a step with no text, a stage that is not in `stages`. Then follow it once with **Show me**
from start to finish. The panel shows a case study's problems under its name, and leaves out the steps it cannot
read.

The full reference is in the plugin's README (`packages/pymetrix-guide/README.md`).
