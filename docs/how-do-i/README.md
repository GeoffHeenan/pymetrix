# How do I…

Task pages: each takes one task from start to finish, in the app and in code, says how to check the result, and
what goes wrong. The rest of the documentation is mostly reference: the [user guide](../guide.md) by feature, the
help topics by dialog or idea, the [course notebooks](../../examples/course) as lessons, and the
[case studies](../case-studies.md) as guided labs. A task page starts from what the reader wants done and points
into those.

This page is the plan: the pages, the template they follow, how readers reach them, and where they live.

## The pages

All fifteen are written:

- [Estimate an equation and test its residuals](../../src/pymetrix/howto/estimate-and-test-an-equation.md)
- [Load data with Browse series (FRED, IMF, ECB, Eurostat, World Bank)](../../src/pymetrix/howto/load-data-with-browse-series.md)
- [Estimate a VAR, then impulse responses and a forecast](../../src/pymetrix/howto/estimate-a-var.md)
- [Bring an IRIS or Dynare model across, then check and solve it](../../src/pymetrix/howto/bring-in-an-iris-or-dynare-model.md)
- [Run a forecast round: baseline, judgement, scenarios, report](../../src/pymetrix/howto/run-a-forecast-round.md)
- [Save, reopen and share a project, and replay its script](../../src/pymetrix/howto/save-and-share-a-project.md)
- [Test for cointegration in a short annual sample](../../src/pymetrix/howto/cointegration-in-a-short-sample.md)
- [Seasonally adjust a series](../../src/pymetrix/howto/seasonally-adjust-a-series.md)
- [Bring an EViews model across, then check and solve it](../../src/pymetrix/howto/bring-in-an-eviews-model.md)
- [Run a script in the app](../../src/pymetrix/howto/run-a-script-in-the-app.md)
- [Make a report](../../src/pymetrix/howto/make-a-report.md)
- [Check a forecast's record (backtest)](../../src/pymetrix/howto/check-a-forecasts-record.md)
- [Solve a large backward-looking model](../../src/pymetrix/howto/solve-a-large-backward-looking-model.md)
- [Reconcile national accounts components to a total](../../src/pymetrix/howto/reconcile-components-to-a-total.md)
- [Translate an EViews program](../../src/pymetrix/howto/translate-an-eviews-program.md) (the `pymetrix-eviews` plugin)

The list, by priority: the tasks most readers meet first, then those of one audience. Audiences: **new** (new to
pymetrix and to econometrics software), **EViews** (experienced EViews users), **IRIS/Dynare** (MATLAB users of
IRIS or Dynare), **code** (people who work in Python first), **forecasters** (forecast rounds in an institution).

| # | Task | Audience | Starts from | Status |
|---|------|----------|-------------|--------|
| 1 | Estimate an equation and test its residuals | new, EViews | Estimate → Equation… | written |
| 2 | Load data with Browse series (FRED, IMF, ECB, Eurostat, World Bank) | new, everyone | File → Browse series… | written |
| 3 | Estimate a VAR, then impulse responses and a forecast | EViews, IRIS/Dynare | Estimate → VARs and local projections → VAR… | written |
| 4 | Bring an IRIS or Dynare model across, then check and solve it | IRIS/Dynare | Model → Import → IRIS file… | written |
| 5 | Run a forecast round: baseline, judgement, scenarios, report | forecasters | Forecast → Make round… | written |
| 6 | Save, reopen and share a project, and replay its script | everyone, code | File → Save project | written |
| 7 | Test for cointegration in a short annual sample | EViews, applied macro | Test → Run test… | written |
| 8 | Seasonally adjust a series | new, EViews | Data → Transform → Seasonal adjustment… | written |
| 9 | Bring an EViews model across, then check and solve it | EViews | Model → Model editor… | written |
| 10 | Run a script in the app | code, EViews (programs) | File → Open script… | written |
| 11 | Make a report | everyone | View → Report | written |
| 12 | Check a forecast's record (backtest) | forecasters | Forecast → Backtest… | written |
| 13 | Solve a large backward-looking model | forecasters, EViews | Model → Solution → Solve model… | written |
| 14 | Reconcile national accounts components to a total | forecasters | Data → Transform → Reconcile components… | written |
| 15 | Translate an EViews program (the `pymetrix-eviews` plugin) | EViews | File → Translate EViews program… | written |

Notes on the list:

- **EViews models apart from IRIS and Dynare (9).** The route differs: the model editor reads an EViews model's
  text or program, and an EViews state-space object comes through Model → Import → EViews state-space object….
  One page for both would branch at every step.
- **Backtest (12) is added** to the candidates: forecasters ask it right after a round, and its pitfalls (the
  sample each forecast may use, revised data) are not in one place now.
- **Solve a large model (13)** was written before the template, as `docs/solving-large-models.md`, with a help
  topic of its own that summarised it (`large-model-solution`). It now has the template's sections and lives beside
  the others; the summary topic is gone, and the help shows the page itself (`howto-solve-a-large-backward-looking-model`).
- **Translate an EViews program (15) is added**, for EViews users with programs. Its task is a plugin's
  (`pymetrix-eviews`), but the page is here, in the core package, with the others: the reader who needs it most is the
  one who does not know the plugin exists, and finds the page in Help → How do I… or with Ctrl+K before installing
  anything. Its front matter names the plugin (`plugin: pymetrix-eviews`), the index says the task needs it, and the
  page starts by saying how to install it. One source and one test module stay: the page's code runs where the plugin
  is installed and is skipped where it is not (the core-only run), its menu paths are checked where the plugin is
  installed (the paragraphs name it), and it links no help topic of the plugin's, so the core help has no link it
  cannot follow. A page in the plugin's package would appear only once the plugin is installed, and would need the
  index to gather pages from plugins.

## The template

Each page, in this order:

1. **`# Title`**: the task as the reader would say it ("Estimate an equation and test its residuals").
2. **Goal**: the first paragraph, no heading. What is done and why, in two or three sentences.
3. **Before you start**: the data or files needed (always ones shipped with pymetrix, or written out on the page,
   so the page runs as written), and the decisions to make first.
4. **In the app**: numbered steps. Menu paths as the menus show them (Estimate → Equation…); fields by the label
   the dialog shows and the value to give (Formula: `dlog(c) ~ dlog(y)`, Standard errors: HAC (Newey-West)); the
   views of the result to read. Choices as the dialog words them, not their codes.
5. **In code**: the same steps as one script, in blocks that run in order. Comments give the numbers the page
   quotes.
6. **Check the result**: what to look at, with the numbers to expect on the page's data, and what to compare with
   another package.
7. **Pitfalls**: one bold sentence each, then why and what to do.
8. **See also**: help topics (as `pymetrix:topic/...` links), other task pages, and the guide's reference sections.

House style as elsewhere: plain, precise, short sentences; no marketing; menu paths written Menu → Item….

Front matter gives each page a `summary` (one line for the index), its `audience` and its `order` in the index. A
file a page's code reads is shown in a fenced block whose info string is `text <file name>`.

## How readers reach the pages

- **Help → How do I…** opens the index: each page with its summary and audience.
- **Help contents** list them in a section of their own, first.
- **Ctrl+K (View → Find anything…)** finds each page by any words of its title: the palette lists every page of
  the help contents (`ui/palette.py`, from `help.contents()`), so a page is found once it is a help topic, with no
  separate registration. "residuals", "cointegration short", "dynare" and "how do i" find them.
- **Help search** finds them by any words of their text, as it finds every topic.
- **Next ▸ (proposed, not built).** A result's Next ▸ menu (`ui/next_steps.py`) offers the next dialogs for its
  type. A second part of the menu, after a separator, could offer the task pages for that type: a table
  `NEXT_PAGES = {"FittedEquation": ("howto-estimate-and-test-an-equation",), "Template": (...), "DynareModel": (...),
  "TestResult": (...)}` and the menu builder in `ui/qt.py` adding "How do I…: <title>" items that open the topic.
  It needs a new kind of entry beside the dialogs (the table today is dialogs only, and its test expects each
  entry to open one), so it is left for a change of its own.
- **Dialog help (proposed).** A dialog's help page could list the task pages that use it, as it lists course
  material (`help.notes_for`): front matter `links: [commands/FitEquation]`, as course notes have.

## One source: Markdown files in the package

The pages are Markdown files in `src/pymetrix/howto/`, and nowhere else:

- **The help shows them as they are.** `pymetrix.howto` reads the files when the help loads and registers each as a
  help topic (`howto-<file name>`) and an index topic (`how-do-i`). For the help, links between pages become help
  links, `pymetrix:` links stay, and links to the repository's documents become their text.
- **GitHub shows them as they are.** This page links to them; `pymetrix:` links show as their text there.
- **The wheel has them.** Files under `src/pymetrix` are in the wheel, as the example datasets are, so the help of
  an installed copy has the pages with no build step. Files under `docs/` are not in the wheel, so pages there
  would need a copy in the package (generated, and checked to match) or a build rule, and two places to look.
- **Not help topics written in Python.** Topics in `help.py` are strings in code: long pages with code blocks are
  hard to edit there, and GitHub would not show them.
- **Not a course.** Course folders (`help.load_course`) are Markdown too, but a course is the user's, loaded from a
  folder and kept with a project; the task pages are pymetrix's own. The front matter reader is shared.

## The tests

`tests/test_how_do_i.py`:

- every page has the template's sections in order, and front matter;
- every page's code runs as written, its blocks in order, with its `text` files written out first, and gives the
  numbers the page quotes;
- the steps in the app give the same results as the code: each step is run as the dialog's command, with the
  values the page names, and each field label the page names is checked against the dialog's own;
- every link resolves (files, anchors' files, help topics), and the help's copy has no link it cannot follow;
- Help → How do I… opens the index, the contents list the pages first, and Ctrl+K finds each page.

`tests/test_docs_menu_paths.py` checks every menu path on the pages and on this plan against the menu bar.

## Found while writing the first pages

- **Check model… did not take an imported model.** Its Equations list showed fitted equations and identities only,
  so an IRIS or Dynare model could not be chosen, though the "Coming from MATLAB/IRIS and Dynare" table sends
  `sstate`, `chksstate`, `steady` and `check` to Model → Check model…. Since 0.1.0a17 it lists every whole model and
  checks an imported one with its own shocks and steady-state guesses, so the page no longer has that pitfall.
- **Help text named a submenu that moved.** Four places (the help of Model → Import → IRIS file…, Model → Template
  model… and Model → Analysis → Kalman filter…, and the shock decomposition topic) said "Model > State space > …";
  the submenu is Model → Analysis now, and they say so. The menu-path test checks paths written with "→" only, so
  paths written with ">" in the help are not checked: writing them with "→", or teaching the test ">", would catch
  the next move.
- **"Mackinnon".** The Critical values choice read "Mackinnon"; it now reads "MacKinnon (asymptotic)", and the other
  choice "Simulated for this sample".

## Found while writing the rest

- **Help text named fields by old labels.** The help on solving models, and the docstrings of Solve model…, Add
  factors… and Import FRB/US XML file…, said "Partial" for Hold only where given, "Errors: sample" for Error draws:
  The data's residuals, "Workers" for Parallel processes, and "Model > Solve model" or "Model > Add factors" for the
  entries now under Model → Solution (as did the Forecast… and Compare scenarios dialogs' help). They use the
  dialogs' own words now. Labels inside help text are not checked by any test; the menu-path test reads "→" only.
- **STL's trend length read "Deterministic terms".** The shared label for fields named `trend` (a VAR's or a unit-root
  test's constant and trend) was given to STL's trend smoother length, under Seasonal adjustment…'s Method settings;
  it reads Trend length now.
- **A dataset loaded from a connector is not kept in the project.** A project keeps the data of files it loaded, but
  File → Load from connector… is run again when the project opens, so the data can come back revised. The pages say
  so, and how to keep the values (write them to a file, or give FRED a vintage); whether a project should keep them is
  a question of its own.
