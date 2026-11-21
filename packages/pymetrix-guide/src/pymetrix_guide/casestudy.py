"""Case studies: Markdown in a course folder, with fenced ``pymetrix-step`` blocks.

    ---
    title: Lab 3. A quarterly forecast round
    summary: A VAR forecast, judgement, revisions, a track record and the report   # optional: under the title
    stages: [Data, Model, Baseline, Judgement, Evaluate, Report]                   # optional: the workflow, in order
    demo: forecast_round          # optional: the demo project it starts from
    level: introductory           # optional
    ---

    ## Overview
    The workflow in general: the first introduction page.

    ## This scenario
    What this case study does, and why: the second.

    ---

    Narrative for the first step...

    ```pymetrix-step
    title: Fit the consumption function
    stage: Model                  # optional: later steps keep it until another is given
    command: FitEquation
    settings: {formula: "c ~ y + c[-1]", name: fit}
    check: object fit is FittedEquation
    why: The purpose of the step, in a sentence.          # optional
    look: What to look at in the result, in a sentence.   # optional
    ```

The text before a block is that step's narrative; the text after the last block closes the case study. When the
text before the first block has ``##`` headings, it is the introduction instead: one page per heading, shown before
step 1, and the first step's own narrative is what follows a thematic break (a ``---`` line) after the last of them.
Without ``##`` headings the text before the first block is the first step's, as it always was. A step's
``command`` is a dialog class name and ``settings`` its fields (a ``{key: value, ...}`` mapping on one line, or
indented ``key: value`` lines below ``settings:``); the dialog registry builds the command. ``check`` is a
declarative condition (``checks.parse_check``): no code runs.

``parse(text)`` never raises for a bad block: it is skipped and reported, with its line, in ``problems``. What a
case study could do better (no introduction, a step with no text) is in ``warnings``, which never stop it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from .checks import Check, CheckError, parse_check

__all__ = ["FENCE", "CaseStudy", "CaseStudyError", "CaseStudyWarning", "IntroPage", "Step", "build_command", "check_case_study",
           "parse", "read"]

FENCE = "pymetrix-step"
STEP_KEYS = ("title", "stage", "command", "settings", "check", "why", "look")
FRONT_KEYS = ("title", "summary", "stages", "demo", "level", "course", "links", "references")


class CaseStudyError(ValueError):
    """A case study with problems: each ``line N: what``."""


class CaseStudyWarning(UserWarning):
    """What a case study could do better (``check_case_study``): no introduction, a step with no text."""


@dataclass(frozen=True)
class IntroPage:
    """A page of the introduction, shown before step 1: a ``##`` section before the first step."""

    title: str
    text: str = ""


@dataclass(frozen=True)
class Step:
    """One step: its narrative (Markdown), title, dialog and settings, and the check that ticks it."""

    title: str
    text: str = ""
    command: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    check: Check | None = None
    line: int = 0  # the line of its ``pymetrix-step`` fence
    stage: str = ""  # the stage of the workflow it belongs to ("Data", "Model", "Judgement"...)
    why: str = ""  # its purpose, in a sentence
    look: str = ""  # what to look at in its result


@dataclass(frozen=True)
class CaseStudy:
    title: str
    steps: tuple[Step, ...]
    demo: str = ""
    level: str = ""
    closing: str = ""  # the text after the last step
    path: Path | None = None
    problems: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)
    intro: tuple[IntroPage, ...] = ()  # the pages before step 1
    summary: str = ""  # one line, under the title
    stages: tuple[str, ...] = ()  # the workflow's stages in order (front matter ``stages``, else the steps' own)
    warnings: tuple[str, ...] = ()  # what it could do better; never stops it


# -- values: a small, safe subset of YAML (strings, numbers, true/false/null, a flat {k: v} mapping) ----------------

def _scalar(text: str) -> Any:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1].replace("\\" + text[0], text[0]) if text[0] == '"' else text[1:-1].replace("''", "'")
    low = text.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", "~", ""):
        return None
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", text):
        return float(text)
    return text


def _text(text: str) -> str:
    """Text as written, without its quotes (front matter, titles: "No" stays a word)."""
    text = text.strip()
    return text[1:-1] if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'" else text


def _split_top(text: str, sep: str) -> list[str]:
    """Split on ``sep`` outside quotes."""
    parts, buf, quote = [], [], ""
    for ch in text:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch == sep:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if quote:
        raise ValueError("a quote is not closed")
    parts.append("".join(buf))
    return parts


def _mapping(text: str) -> dict[str, Any]:
    """``{a: 1, b: "x, y"}`` as a dict."""
    text = text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise ValueError("settings are a mapping, {key: value, ...}, or indented key: value lines")
    out: dict[str, Any] = {}
    body = text[1:-1].strip()
    if not body:
        return out
    for item in _split_top(body, ","):
        key, sep, value = item.partition(":")
        key = key.strip()
        if not sep or not re.fullmatch(r"[A-Za-z_]\w*", key):
            raise ValueError(f"{item.strip()!r} is not key: value")
        if value.strip().startswith(("{", "[")):
            raise ValueError(f"{key}: settings are plain values (text, numbers, true or false), not nested")
        out[key] = _scalar(value)
    return out


def _front_matter(lines: list[str]) -> tuple[dict[str, Any], int]:
    """The ``---`` block at the top: (keys, the index of the first line after it)."""
    if not lines or lines[0].strip() != "---":
        return {}, 0
    meta: dict[str, Any] = {}
    key = ""
    for i in range(1, len(lines)):
        line = lines[i]
        if line.strip() == "---":
            return meta, i + 1
        if line.lstrip().startswith("- ") and key:
            if not isinstance(meta.get(key), list):
                meta[key] = [meta[key]] if meta.get(key) else []
            meta[key].append(_text(line.lstrip()[2:]))
        elif ":" in line:
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if value.startswith("[") and value.endswith("]"):
                try:
                    meta[key] = [_text(v) for v in _split_top(value[1:-1], ",") if v.strip()]
                except ValueError:  # a quote not closed: kept as written
                    meta[key] = value
            else:
                meta[key] = _text(value)
    return {}, 0  # not closed: not front matter


# -- steps -------------------------------------------------------------------------------------------------------------

def _step_block(body: list[tuple[int, str]], fence_line: int, text: str) -> tuple[Step | None, list[str]]:
    """A ``pymetrix-step`` block's ``key: value`` lines as a Step, or None and its problems."""
    problems: list[str] = []
    values: dict[str, Any] = {}
    where: dict[str, int] = {}  # each key's line
    bad_settings = False
    settings: dict[str, Any] = {}
    in_settings = False
    for n, raw in body:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indented = raw[:1] in (" ", "\t")
        if in_settings and indented:
            key, sep, value = raw.strip().partition(":")
            if not sep or not re.fullmatch(r"[A-Za-z_]\w*", key.strip()):
                problems.append(f"line {n}: {raw.strip()!r} is not key: value")
                bad_settings = True
            else:
                try:
                    _split_top(value, ",")  # a closed quote
                    settings[key.strip()] = _scalar(value)
                except ValueError as exc:
                    problems.append(f"line {n}: {exc}")
                    bad_settings = True
            continue
        in_settings = False
        key, sep, value = raw.partition(":")
        key = key.strip()
        if not sep or indented:
            problems.append(f"line {n}: {raw.strip()!r} is not key: value")
            continue
        if key not in STEP_KEYS:
            problems.append(f"line {n}: unknown key {key!r} (a step has {', '.join(STEP_KEYS)})")
            continue
        if key in values:
            problems.append(f"line {n}: {key} given twice")
            continue
        value = value.strip()
        where[key] = n
        if key == "settings":
            if not value:
                in_settings = True
                values[key] = True
                continue
            try:
                settings = _mapping(value)
            except ValueError as exc:
                problems.append(f"line {n}: {exc}")
                bad_settings = True
            values[key] = True
        elif key == "check":
            try:
                values[key] = parse_check(value)
            except CheckError as exc:
                problems.append(f"line {n}: {exc}")
                values[key] = None
        else:
            values[key] = _text(value)
    title = values.get("title", "")
    if not title:
        problems.append(f"line {fence_line}: the step has no title")
    command = values.get("command", "")
    if settings and not command:
        problems.append(f"line {where['settings']}: settings without a command")
    if not command and "check" not in values:
        problems.append(f"line {fence_line}: a step needs a command, a check or both")
    if command and not bad_settings:
        try:
            build_command(command, settings)
        except CaseStudyError as exc:
            at = where.get("settings", where["command"]) if str(exc).startswith("settings:") else where["command"]
            problems.append(f"line {at}: {exc}")
    if problems:
        return None, problems
    return Step(title, text.strip(), command, settings, values.get("check"), fence_line, stage=values.get("stage", ""),
                why=values.get("why", ""), look=values.get("look", "")), []


_FENCE_LINE = re.compile(r"^\s*(`{3,}|~{3,})")
_H2 = re.compile(r"^##\s+(.+?)\s*#*\s*$")
_RULE = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")


def _introduction(lines: list[str]) -> tuple[list[IntroPage], str]:
    """The text before the first step: its ``##`` sections as introduction pages, and the first step's own text (after
    a thematic break that follows the last section). Without ``##`` headings, all of it is the first step's text."""
    heads: list[int] = []
    rules: list[int] = []
    fence = ""
    for i, line in enumerate(lines):
        m = _FENCE_LINE.match(line)
        if fence:
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                fence = ""
            continue
        if m:
            fence = m.group(1)
        elif _H2.match(line):
            heads.append(i)
        elif _RULE.match(line) and (i == 0 or not lines[i - 1].strip()):  # after a blank line: not a heading's underline
            rules.append(i)
    if not heads:
        return [], "\n".join(lines)
    after = [r for r in rules if r > heads[-1]]
    end = after[-1] if after else len(lines)
    rest = "\n".join(lines[end + 1:]) if after else ""
    pages: list[IntroPage] = []
    lead = [ln for ln in lines[:heads[0]] if not re.match(r"^#\s", ln)]  # the title heading is the panel's already
    if "\n".join(lead).strip():
        pages.append(IntroPage("Introduction", "\n".join(lead).strip()))
    for k, at in enumerate(heads):
        stop = heads[k + 1] if k + 1 < len(heads) else end
        pages.append(IntroPage(_H2.match(lines[at]).group(1).strip(), "\n".join(lines[at + 1:stop]).strip()))
    return pages, rest


def parse(text: str, path: str | Path | None = None) -> CaseStudy:
    """A case study from its Markdown. Bad blocks are left out and reported (``problems``), with their line."""
    lines = text.splitlines()
    meta, start = _front_matter(lines)
    problems = [f"line 1: unknown front-matter key {k!r}" for k in meta if k not in FRONT_KEYS]
    steps: list[Step] = []
    narrative: list[str] = []
    intro: list[IntroPage] = []
    i = start
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\s*)(`{3,}|~{3,})\s*(\S*)", line)
        if not m:
            narrative.append(line)
            i += 1
            continue
        fence, info = m.group(2), m.group(3)
        j = i + 1
        while j < len(lines) and not lines[j].strip().startswith(fence):
            j += 1
        if info != FENCE:  # an ordinary code block belongs to the narrative
            narrative.extend(lines[i:j + 1])
            i = j + 1
            continue
        if j >= len(lines):
            problems.append(f"line {i + 1}: the {FENCE} block is not closed")
            break
        if not steps and not intro:  # the text before the first step: an introduction, when it has ## sections
            intro, first = _introduction(narrative)
            narrative = first.splitlines()
        step, bad = _step_block([(k + 1, lines[k]) for k in range(i + 1, j)], i + 1, "\n".join(narrative))
        problems += bad
        if step is not None:
            if not step.stage and steps and steps[-1].stage:  # a stage lasts until the next is named
                step = replace(step, stage=steps[-1].stage)
            steps.append(step)
            narrative = []
        i = j + 1
    if not steps and not problems:
        problems.append(f"no {FENCE} blocks: this is not a case study")
    heading = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    title = str(meta.get("title") or (heading.group(1).strip() if heading else (Path(path).stem if path else "Case study")))
    demo = str(meta.get("demo") or "")
    if demo:
        from pymetrix import demos

        if demo not in demos():
            problems.append(f"line 1: no demo {demo!r}; demos: {', '.join(demos())}")
            demo = ""
    named = meta.get("stages") or []
    named = [str(s).strip() for s in (named if isinstance(named, list) else str(named).split(",")) if str(s).strip()]
    stages = tuple(named) or tuple(dict.fromkeys(s.stage for s in steps if s.stage))
    warnings: list[str] = []
    if steps and not intro:
        warnings.append("no introduction: put ## sections before the first step (the workflow in general, then this case study)")
    for n, st in enumerate(steps, 1):
        if not st.text and not st.why:
            warnings.append(f"line {st.line}: step {n} ({st.title}) has no text: say what to do and why")
        if st.stage and st.stage not in stages:
            warnings.append(f"line {st.line}: step {n} has stage {st.stage!r}, not one of the stages ({', '.join(stages)})")
        elif stages and not st.stage:
            warnings.append(f"line {st.line}: step {n} ({st.title}) has no stage")
    return CaseStudy(title, tuple(steps), demo, str(meta.get("level") or ""), "\n".join(narrative).strip(),
                     Path(path) if path else None, tuple(problems), meta, tuple(intro), str(meta.get("summary") or ""),
                     stages, tuple(warnings))


def read(path: str | Path) -> CaseStudy:
    """A case study from its file."""
    path = Path(path)
    return parse(path.read_text(encoding="utf-8"), path)


def check_case_study(path: str | Path) -> CaseStudy:
    """Read a case study and raise ``CaseStudyError`` listing its problems, if any: for a lecturer, before handing it out.
    What it could do better (no introduction, a step with no text) is said as a ``CaseStudyWarning`` each."""
    import warnings

    study = read(path)
    if study.problems:
        raise CaseStudyError(f"{Path(path).name}: " + "; ".join(study.problems))
    for warning in study.warnings:
        warnings.warn(f"{Path(path).name}: {warning}", CaseStudyWarning, stacklevel=2)
    return study


# -- building a step's command from the dialog registry ------------------------------------------------------------------

def _coerce(value: Any, hint: Any) -> Any:
    """A setting as its field's type, where the case study wrote it plainly (``lags: 2`` for a text field)."""
    from typing import Annotated, Union, get_args, get_origin

    if get_origin(hint) is Annotated:
        hint = get_args(hint)[0]
    options = get_args(hint) if get_origin(hint) in (Union, type(int | None)) else (hint,)
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if str in options and int not in options and float not in options:
        return str(value)
    if isinstance(value, int) and float in options and int not in options:
        return float(value)
    return value


def build_command(name: str, settings: dict[str, Any] | None = None) -> Any:
    """The dialog command ``name`` (a class name in the registry) with ``settings``; ``CaseStudyError`` says what is
    wrong (no such dialog, a field it does not have, one it needs)."""
    from typing import get_type_hints

    from pymetrix.commands import COMMANDS

    settings = dict(settings or {})
    info = COMMANDS.get(name)
    if info is None:
        near = [n for n in COMMANDS if n.lower() == name.lower()]
        raise CaseStudyError(f"no dialog {name!r}" + (f" (did you mean {near[0]}?)" if near else ""))
    cls = info.cls
    names = [f.name for f in fields(cls) if f.init]
    unknown = [k for k in settings if k not in names]
    if unknown:
        raise CaseStudyError(f"settings: {name} has no field {', '.join(map(repr, unknown))} (it has {', '.join(names)})")
    try:
        hints = get_type_hints(cls, include_extras=True)
    except Exception:  # noqa: BLE001 - coercion is a convenience
        hints = {}
    values = {k: _coerce(v, hints.get(k, Any)) for k, v in settings.items()}
    try:
        return cls(**values)
    except TypeError as exc:
        missing = re.findall(r"'(\w+)'", str(exc)) if "missing" in str(exc) else []
        raise CaseStudyError(f"settings: {name} needs {', '.join(missing)}" if missing else f"settings: {exc}") from None
    except Exception as exc:  # noqa: BLE001 - the dialog's own validation
        raise CaseStudyError(f"settings: {exc}") from None
