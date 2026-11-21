"""The case-study format: steps, front matter, bad blocks reported with their line; and the checks."""

import warnings

import pytest
from pymetrix_guide import (
    CaseStudyError,
    CaseStudyWarning,
    IntroPage,
    CheckError,
    build_command,
    case_studies,
    check_case_study,
    course_folders,
    parse,
    parse_check,
    read,
)

from pymetrix import Dataset
from pymetrix.commands import FitEquation, GenerateSeries, Session

STUDY = '''---
title: Consumption, guided
demo: consumption
level: introductory
course: ECON 101
links: [commands/FitEquation]
---

# Consumption

Some narrative, with code that is not a step:

```python
print("not a step")
```

```pymetrix-step
title: Fit the consumption function
command: FitEquation
settings: {formula: "c ~ y + c[-1]", name: fit}
check: object fit is FittedEquation
```

Now a series.

~~~pymetrix-step
title: Log output
command: GenerateSeries
settings:
  series: ly
  formula: "log(y)"
check: series ly in data and object data is Dataset
~~~

```pymetrix-step
title: Look at the residuals
check: object fit exists
```

The end.
'''


def test_steps_front_matter_and_narrative():
    study = parse(STUDY, "consumption.md")
    assert study.problems == ()
    assert (study.title, study.demo, study.level) == ("Consumption, guided", "consumption", "introductory")
    assert [s.title for s in study.steps] == ["Fit the consumption function", "Log output", "Look at the residuals"]
    first, second, third = study.steps
    assert first.command == "FitEquation" and first.settings == {"formula": "c ~ y + c[-1]", "name": "fit"}
    assert "Some narrative" in first.text and 'print("not a step")' in first.text and "# Consumption" in first.text
    assert second.settings == {"series": "ly", "formula": "log(y)"} and second.text == "Now a series."
    assert [c.kind for c in second.check.conditions] == ["series", "is"]
    assert third.command == "" and third.check.describe() == "fit exists"
    assert study.closing == "The end." and first.line == 17
    # no ## sections before the first step: its text is the first step's, as before, and that is a warning
    assert study.intro == () and study.stages == () and study.summary == ""
    assert study.warnings == ("no introduction: put ## sections before the first step (the workflow in general, then this case "
                              "study)", "line 35: step 3 (Look at the residuals) has no text: say what to do and why")


INTRODUCED = '''---
title: Consumption, guided
summary: A consumption function, static and dynamic
stages: [Data, Model, Evaluate]
---

# Consumption, guided

A lead paragraph.

## Overview

How an applied study goes.

```python
## not a heading: code
```

## This scenario

The data and the question.

***

Load the data first.

```pymetrix-step
title: Log output
stage: Data
command: GenerateSeries
settings: {series: ly, formula: "log(y)"}
check: series ly in data
why: The model is in logs.
look: ly is in data.
```

```pymetrix-step
title: Fit
stage: Model
command: FitEquation
settings: {formula: "c ~ y", name: fit}
why: The static function first.
```

```pymetrix-step
title: Fit again
command: FitEquation
settings: {formula: "c ~ y + c[-1]", name: dynamic}
```

```pymetrix-step
title: Elsewhere
stage: Forecast
check: object dynamic exists
```

Done.
'''


def test_an_introduction_summary_and_stages():
    study = parse(INTRODUCED)
    assert study.problems == () and study.summary == "A consumption function, static and dynamic"
    assert study.intro == (IntroPage("Introduction", "A lead paragraph."),
                           IntroPage("Overview", "How an applied study goes.\n\n```python\n## not a heading: code\n```"),
                           IntroPage("This scenario", "The data and the question."))
    first, second, third, fourth = study.steps
    assert first.text == "Load the data first." and (first.why, first.look) == ("The model is in logs.", "ly is in data.")
    assert [s.stage for s in study.steps] == ["Data", "Model", "Model", "Forecast"]  # a stage lasts until the next
    assert study.stages == ("Data", "Model", "Evaluate") and study.closing == "Done."
    assert study.warnings == ("line 45: step 3 (Fit again) has no text: say what to do and why",
                              "line 51: step 4 (Elsewhere) has no text: say what to do and why",
                              "line 51: step 4 has stage 'Forecast', not one of the stages (Data, Model, Evaluate)")
    # without front-matter stages, the steps' own, in order; a heading's underline is not the break
    plain = parse("## Overview\n\nAll of it.\n---\n\n```pymetrix-step\ntitle: A\nstage: Data\ncheck: object x exists\n```\n")
    assert plain.stages == ("Data",) and plain.intro == (IntroPage("Overview", "All of it.\n---"),) and plain.steps[0].text == ""
    assert plain.warnings == ("line 6: step 1 (A) has no text: say what to do and why",)


def test_check_case_study_warns_and_does_not_raise(tmp_path):
    (tmp_path / "plain.md").write_text(STUDY, encoding="utf-8")
    with pytest.warns(CaseStudyWarning) as said:
        study = check_case_study(tmp_path / "plain.md")
    assert len(study.steps) == 3 and [str(w.message) for w in said] == [f"plain.md: {w}" for w in study.warnings]
    assert str(said[0].message).startswith("plain.md: no introduction") and len(said) == 2


def test_settings_build_the_dialogs_command_through_the_registry():
    assert build_command("FitEquation", {"formula": "c ~ y", "name": "fit"}) == FitEquation("c ~ y", name="fit")
    from pymetrix.commands import FitVAR

    cmd = build_command("FitVAR", {"variables": "ly, lc", "lags": 2, "tightness": 1})  # plain values, as the field's type
    assert cmd.lags == "2" and cmd.tightness == 1.0 and isinstance(cmd, FitVAR)
    with pytest.raises(CaseStudyError, match="no dialog 'FitEquations'"):
        build_command("FitEquations")
    with pytest.raises(CaseStudyError, match="did you mean FitEquation"):
        build_command("fitequation")
    with pytest.raises(CaseStudyError, match="has no field 'formulae'"):
        build_command("FitEquation", {"formulae": "c ~ y"})
    with pytest.raises(CaseStudyError, match="needs formula"):
        build_command("FitEquation", {})


BAD = '''---
title: Bad
demo: nowhere
colour: blue
---

```pymetrix-step
title: Fine
command: GenerateSeries
settings: {series: z, formula: "y + 1"}
check: series z in data
```

```pymetrix-step
title: An unknown dialog
command: FitEquations
```

```pymetrix-step
title: A bad check
command: FitEquation
settings: {formula: "c ~ y"}
check: __import__('os').system('rm -rf /')
```

```pymetrix-step
command: FitEquation
settings: {formula: "c ~ y", colour: red}
unknown: 1
```

```pymetrix-step
title: Settings that do not close
command: FitEquation
settings: {formula: "c ~ y}
```

```pymetrix-step
title: Not closed
'''


def test_bad_blocks_are_reported_with_their_line_and_left_out():
    study = parse(BAD)
    assert [s.title for s in study.steps] == ["Fine"]
    problems = "\n".join(study.problems)
    assert "line 1: unknown front-matter key 'colour'" in problems
    assert "line 1: no demo 'nowhere'" in problems and study.demo == ""
    assert "line 16: no dialog 'FitEquations'" in problems
    assert "line 23: \"__import__('os').system('rm -rf /')\" is not a check" in problems
    assert "line 29: unknown key 'unknown'" in problems and "line 26: the step has no title" in problems
    assert "line 28: settings: FitEquation has no field 'colour'" in problems
    assert "line 35: a quote is not closed" in problems and "needs formula" not in problems
    assert "line 38: the pymetrix-step block is not closed" in problems
    with pytest.raises(CaseStudyError, match="line 16"):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "bad.md").write_text(BAD, encoding="utf-8")
            check_case_study(Path(d) / "bad.md")


def test_text_stays_text_and_nothing_raises():
    study = parse('---\ntitle: "No"\nlinks: [a, "b\n---\n```pymetrix-step\ntitle: Yes\ncheck: object x exists\n```\n')
    assert study.title == "No" and study.steps[0].title == "Yes" and study.problems == ()
    for text in ("", "---", "---\n---", "```pymetrix-step", "```pymetrix-step\n```", "```pymetrix-step\n:\n  :\n```",
                 "```pymetrix-step\nsettings:\n  a: 'b\n```", "```pymetrix-step\ncommand: FitEquation\nsettings: {a: {b: 1}}\n```"):
        assert parse(text).problems  # reported, never raised


def test_a_note_without_steps_is_not_a_case_study():
    assert parse("# Notes\n\nNo steps here.\n").problems == ("no pymetrix-step blocks: this is not a case study",)


def test_the_checks():
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    assert parse_check("object data exists").holds(s) and not parse_check("object fit exists").holds(s)
    assert parse_check("object data is Dataset").holds(s) and not parse_check("object data is FittedEquation").holds(s)
    assert parse_check("series y in data").holds(s) and not parse_check("series ly in data").holds(s)
    assert not parse_check("series y in nowhere").holds(s)
    assert not parse_check("step GenerateSeries ran").holds(s)
    s.run(GenerateSeries("ly", "log(y)"))
    s.run(FitEquation("c ~ y", name="fit"))
    assert parse_check("series ly in data and object fit is FittedEquation").holds(s)
    assert parse_check("object fit is an Estimate").holds(s) is any(c.__name__ == "Estimate" for c in type(s["fit"]).__mro__)
    assert parse_check("step GenerateSeries ran on data").holds(s) and not parse_check("step GenerateSeries ran on fit").holds(s)
    assert parse_check("step FitEquation ran").holds(s)
    s.undo()
    assert not parse_check("object fit exists").holds(s) and not parse_check("step FitEquation ran").holds(s)
    assert not parse_check("object data exists").holds(None)
    assert parse_check("object  fit\n exists").describe() == "fit exists"
    for bad in ("", "object fit", "object fit is", "len(data) > 3", "object fit exists or object x exists", "series y in data[0]"):
        with pytest.raises(CheckError, match="a check is"):
            parse_check(bad)


def test_the_plugin_ships_lab_3_as_a_course():
    from pymetrix.testing import check_course

    (folder,) = course_folders()
    course = check_course(folder)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CaseStudyWarning)  # an introduction, and every step with its text
        study = check_case_study(folder / "03_forecast_round_guided.md")
    assert len(study.steps) == 19 and study.demo == "" and course.name == "Case studies (pymetrix-guide)"
    assert [p.title for p in study.intro] == ["Overview: the forecast workflow", "How the app keeps the record", "This scenario",
                                              "Using the guide"]
    assert study.summary and study.stages == ("Data", "Model", "Baseline", "Judgement", "Revisions", "Evaluate", "Report")
    assert all(s.stage and s.why and s.look and s.text for s in study.steps)
    assert "## " not in study.steps[0].text and "Where to go next" in study.closing
    assert any(s.path == study.path for s in case_studies())
    assert read(study.path).steps == study.steps
