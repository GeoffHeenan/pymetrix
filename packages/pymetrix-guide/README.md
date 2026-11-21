# pymetrix-guide

Guided case studies for pymetrix, as a plugin: a trainee follows a workflow step by step in the desktop app,
through the app's real dialogs, and each step is checked as they go. Experimental.

```bash
pip install -e packages/pymetrix-guide          # from the repository (not on PyPI yet)
```

Once installed, the app has a **Guide** panel (View → Guide). It lists the case studies in the loaded course
material: the plugin's own, folders added with **Help → Course material…** or `PYMETRIX_COURSES`, courses
published as packages, and the courses saved with the open project. For the chosen case study it shows:

- its summary under the title, and its introduction first: a page for each `##` section before the first step
  (for Lab 3, an overview of the forecast workflow, how the app keeps the record, this scenario, and how to use the
  guide);
- one step at a time: its text, why it is done and what to look for in its result, and progress
  ("Stage: Judgement — step 8 of 19 · 7 done"), with the stages of the workflow as a strip, the current one in bold;
- **Open…**: the step's dialog, filled in. The trainee looks at it, changes it or not, and presses OK;
- **Show me**: runs the step as written;
- a tick on each step whose check passes. Ticks follow the history, so **Undo** takes one back and **Redo** gives it
  again;
- **Back** and **Next**, through the introduction and then the steps, and **Start over**: the introduction's first
  page, and the case study's demo project again;
- for a case study that starts from a demo project (`demo:`), an offer to open it while another project is open.
  The demo opens as with **File → Open demo project…**: the app asks to save unsaved work first.

The steps are ordinary steps of the session: the History, **File → Export script**, the saved project and the
lab book (`pymetrix-labbook`) all record them. The guide keeps no record of its own.

The plugin ships Lab 3 of the example course, the quarterly forecast round (`examples/course/03_forecast_round`), as
its first case study: an introduction of four pages, then 19 steps in seven stages, from loading the data to the
report. It is a course in the `pymetrix.courses`
group, *Case studies (pymetrix-guide)*, so it is in the help as well.

## The format

A case study is a Markdown file in a course folder. Narrative text goes between fenced blocks whose info string is
`pymetrix-step`:

````markdown
---
title: Lecture 1, guided. A consumption function
summary: A dynamic consumption function and its long-run propensity   # optional
stages: [Data, Model, Interpret]   # optional: the workflow's stages, in order
demo: consumption              # optional: the demo project it starts from
level: introductory            # optional
course: ECON 101               # the usual course front matter
links: [commands/FitEquation]
---

# A consumption function

## Overview

The workflow in general, in the setting of the example: the first page of the introduction.

## This scenario

The data, the model and why, what is compared: the second page.

---

Text before a step is that step's text: what to do.

```pymetrix-step
title: Fit the consumption function
stage: Model
command: FitEquation
settings: {formula: "c ~ y + c[-1]", name: fit}
check: object fit is FittedEquation
why: The dynamic version: consumption adjusts to income over time.
look: The coefficient on c[-1]: the larger it is, the slower the adjustment.
```

Text after the last step closes the case study: a recap, and where to go next.
````

### The introduction

When the text before the first step has `##` headings, it is the introduction: one page per `##` section, shown
before step 1 (text before the first `##`, other than the `#` title, is a page "Introduction"). The first step's
own text follows a line `---`, after a blank line, below the last section; without it the first step has no text
of its own. Without any `##` heading, the text before the first step is that step's text, as in earlier versions.

### Front matter

| Key | Meaning |
|---|---|
| `title` | The case study's title in the panel (else its first `#` heading, else the file name). |
| `summary` | One line, shown under the title when the case study is chosen. |
| `stages` | The stages of the workflow, in order (`[Data, Model, Judgement]`): the panel's strip. Else the steps' own stages, in order. |
| `demo` | A demo project it starts from (`pymetrix.demos()`): the panel offers to open it, and Start over opens it again. |
| `level` | Free text, e.g. introductory. |
| `course`, `links`, `references` | As for any course note (see pymetrix's guide, "Help and course material"). |

### A step

| Key | Meaning |
|---|---|
| `title` | Required. The step's name in the list. |
| `stage` | The stage of the workflow it belongs to; a step without one is in the stage of the step before. |
| `command` | A dialog's class name, as in the registry (`pymetrix.commands.COMMANDS`): `FitEquation`, `GenerateSeries`, `FitVAR`… |
| `settings` | The dialog's fields: `{field: value, ...}` on one line, or indented `field: value` lines below `settings:`. Values are text (quoted when they hold a comma, colon or brace), numbers, `true` or `false`. A number given for a text field is used as text. |
| `check` | What ticks the step (below). Without one the step never ticks; the trainee presses Next. |
| `why` | Its purpose, in a sentence: shown as "Why this step". |
| `look` | What to look at in its result, in a sentence: shown as "What to look for". |

A step needs a `command`, a `check` or both. Lines starting with `#` are comments.

### Checks

A check is a declarative condition, never code. The grammar is:

| Check | Passes when |
|---|---|
| `object NAME exists` | an object called NAME is in the workspace |
| `object NAME is TYPE` | ... and its class, or a class it derives from, is called TYPE (`FittedEquation`, `Dataset`, `ForecastRound`) |
| `series NAME in DATASET` | the dataset DATASET has a series NAME |
| `step DIALOG ran` | a step of that dialog is in the history and ran without error (for steps that change an object in place) |
| `step DIALOG ran on NAME` | ... and read or wrote NAME |

Join conditions with `and`: `series ly in data and series lc in data`.

### Checking a case study

A block that cannot be read is left out and reported with its line: in the panel, under the case study's name,
and by `check_case_study`, which raises on any problem. What would make it better is a warning
(`CaseStudyWarning`, and `study.warnings`), never an error: no introduction, a step with neither text nor `why`,
a stage not in `stages`, or a step with no stage when others have one.

```python
from pymetrix_guide import check_case_study, read

check_case_study("my_course/week1_guided.md")    # CaseStudyError: week1_guided.md: line 23: 'len(data) > 3' is not a check ...
study = read("my_course/week1_guided.md")        # study.intro, study.steps, study.stages, study.problems, study.warnings
```

It checks that each dialog exists, that the settings are its fields and that it can be built from them, and each
check against the grammar. `pymetrix.testing.check_course(folder)` checks the folder as course material.

## Limits

- The panel opens a case study's demo through the app's interface (`AppInterface.open_demo`, extension API 1.7):
  the app asks to save unsaved work first, and the trainee can cancel. With an older pymetrix (API 1.6 or earlier) the
  panel says where to find the demo instead (**File → Open demo project…**, **File → Reload demo project**).
- Checks look at the workspace and the history, not at values: "the coefficient is about 0.9" is not a check.
- A step that writes files (Save round, Export report) writes them where the app's working folder is.

BSD-3-Clause.
