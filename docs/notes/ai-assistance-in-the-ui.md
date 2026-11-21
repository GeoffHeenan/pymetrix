# AI Assistance In The pymetrix UI

Draft note for the pymetrix repository, October 2026.

> **Status (11 October 2026): a design direction, not built.** pymetrix has no AI assistant yet.
> The principle is adopted: AI sits above the command/session layer and proposes commands that
> run, after approval, through the same engine, history and project replay as every dialog.
> Much of the "AI-ready plumbing" of Phase A already exists:
>
> - command schemas: the command registry with stable IDs, and `form_spec` field specs with help;
> - result summaries: registered views (`view_sections`) of tables and figures;
> - help, defaults and validation: the help topics and `docs/validation.md`;
> - generated Python for every step (`Session.script()`), and project trust for typed code.
>
> Still to build for Phase A: the context-bundle format, the redaction policy and the structured
> proposal format.
>
> **18 October 2026:** the first form of the principle is built: an [MCP server](mcp-server.md). An assistant runs
> commands in a session, the client asks the user before each one, and the session exports as a script and
> a project.

## Summary

pymetrix should support AI assistance early, but in a constrained, human-in-the-loop form.

The strongest open-source analogues are not autonomous agents that freely modify projects. They are specialized workbench UIs where AI accelerates a structured workflow and the human expert reviews, edits and accepts the result.

For pymetrix, the right first target is:

> AI proposes commands, explanations, diagnostics and report text. pymetrix shows the proposed command and generated Python. The user approves before anything runs.

Agentic AI should come later, after project trust, audit, replay, undo and validation systems are mature.

## Relevant Analogues

### CVAT

CVAT is an open-source image and video annotation workbench. Its AI tools support automatic and semi-automatic detection, segmentation and tracking, including Segment Anything-style workflows ([CVAT overview](https://docs.cvat.ai/docs/getting_started/overview/), [CVAT AI tools](https://docs.cvat.ai/docs/annotation/auto-annotation/ai-tools/)).

Lesson for pymetrix:

- AI accelerates a domain workflow.
- The user remains in the annotation UI.
- AI output is reviewed and edited by a human.
- AI assistance is attached to concrete objects, not a free-floating chatbot.

### Label Studio

Label Studio is an open-source data labeling platform. It integrates with machine-learning backends that supply predictions or pre-labels and can support active-learning workflows ([Label Studio documentation](https://labelstud.io/guide/get_started)).

Lesson for pymetrix:

- AI assistance can be modular and pluggable.
- The core application does not need to own every model.
- Predictions should enter the workflow as proposed labels or actions, not invisible state changes.

### Jupyter AI

Jupyter AI is an open-source JupyterLab extension that connects AI agents to computational notebooks ([Project Jupyter AI](https://jupyter.org/ai), [Jupyter AI GitHub](https://github.com/jupyterlab/jupyter-ai)).

Lesson for pymetrix:

- AI belongs naturally in computational workflows.
- Notebook context is valuable but must be scoped.
- The assistant should help explain, generate and modify computational artifacts in a visible way.

### Continue

Continue is an open-source AI code assistant for VS Code and JetBrains, with support for custom agents ([Continue documentation](https://docs.continue.dev/?trk=public_profile__posts-text)).

Lesson for pymetrix:

- AI can sit beside an expert workflow as a companion panel.
- Users value provider choice, local/cloud flexibility and control over context.
- The assistant is most useful when it understands project structure and can produce editable artifacts.

### ilastik

ilastik is an open-source interactive machine-learning workbench for biological image analysis. Users paint sparse labels and immediately see predictions; no machine-learning expertise is required ([ilastik](https://www.ilastik.org/)).

Lesson for pymetrix:

- The best AI assistance can make complex methods usable without hiding them.
- Immediate feedback is powerful.
- The workflow remains structured and reproducible.

### FiftyOne

FiftyOne is an open-source tool for visual dataset curation, annotation, model evaluation and data-quality workflows. It now includes agent/MCP-oriented tools for taking actions on datasets ([FiftyOne GitHub](https://github.com/voxel51/fiftyone), [FiftyOne agents](https://docs.voxel51.com/agents/index.html)).

Lesson for pymetrix:

- AI assistance can be built around data inspection, curation and workflow actions.
- Rich object summaries and app state make assistants more useful.
- Dataset privacy and user control need explicit handling.

### QuPath And AI Extensions

QuPath is an open-source digital pathology workbench. AI extensions such as WSInfer bring deep-learning inference into an interactive pathology UI, where users can inspect and curate results ([WSInfer and QuPath](https://www.nature.com/articles/s41698-024-00499-9.pdf)).

Lesson for pymetrix:

- AI is most credible when embedded in validated domain workflows.
- Expert review remains central, especially in high-stakes domains.

## Main Design Lesson

The successful pattern is:

1. The application has structured domain objects.
2. The assistant receives a limited, explicit context.
3. The assistant proposes an action or explanation.
4. The user reviews and approves.
5. The application applies the action through its ordinary engine.
6. The result is auditable and reproducible.

For pymetrix, the structured objects are datasets, equations, fitted results, models, scenarios, forecast rounds, reports, command history, help pages and provenance.

## Recommended Initial Scope

The first AI assistant should support:

- explain the selected result;
- suggest diagnostics for the selected object;
- explain an error message;
- search and summarize help, defaults and validation notes;
- pre-fill command forms from natural language;
- propose Python code equivalent to a guided action;
- draft report text from selected results;
- summarize provenance and reproducibility checks;
- suggest next steps in a teaching or forecasting workflow.

The assistant should not initially:

- run arbitrary Python;
- modify data or projects without approval;
- choose complex methods silently;
- send raw data, model equations or confidential forecasts to a cloud provider by default;
- autonomously perform multi-step analyses;
- download external data without explicit user action.

## Architecture

AI assistance should sit above the command/session layer:

```text
UI
  -> AI assistant panel
      -> context bundle
      -> help and validation search
      -> command proposal generator
      -> user approval
  -> command/session engine
      -> ordinary commands
      -> generated Python
      -> history, provenance and project replay
```

The AI layer must not have a private way to mutate state. It should propose commands that pass through the same validation, execution, history and project machinery as every other UI action.

## Context Bundles

The assistant should receive an explicit context bundle rather than unrestricted access to the session.

Example:

```json
{
  "mode": "metadata_only",
  "selected_object": {
    "type": "FittedEquation",
    "name": "consumption_fit",
    "sample": "2000Q1-2019Q4",
    "stats": {
      "nobs": 80,
      "r2": 0.93
    }
  },
  "available_commands": ["FitEquation", "RunTest", "ForecastRound"],
  "history_summary": "...",
  "help_snippets": ["..."],
  "redactions": ["raw_data", "formula", "file_paths"]
}
```

The context bundle should be generated by pymetrix, not assembled ad hoc by prompts.

## Structured Outputs

The assistant should return structured proposals:

```json
{
  "type": "command_proposal",
  "command": "FitEquation",
  "fields": {
    "formula": "c ~ y + c[-1]",
    "method": "OLS",
    "options": "cov='hac'"
  },
  "code": "fit = Equation('c ~ y + c[-1]').fit(data, method=OLS(cov='hac'))",
  "explanation": "Estimates consumption on income and lagged consumption with HAC standard errors.",
  "risks": ["Check serial correlation and sample endpoints before relying on inference."]
}
```

The UI should show:

- the proposed action;
- the generated Python;
- what context was used;
- what, if anything, would be sent to an external model;
- buttons to run, edit or cancel.

## Confidentiality Modes

pymetrix should support project-level AI modes:

1. `off`: no AI calls.
2. `local_help`: local help search and deterministic suggestions only.
3. `metadata_only`: cloud AI can see sanitized object summaries, command schemas and help snippets, but not raw data, formulas, confidential file paths, report drafts or model equations unless explicitly approved.
4. `full_context`: selected data, code, result tables or report text may be sent only after explicit per-action approval.
5. `local_model`: use a local model endpoint where available.

Default for professional users should be `off` or `metadata_only`, depending on installation profile. Full-context cloud mode should never be silent.

## Sequencing

### Phase A: AI-Ready Plumbing

Build this before or alongside the first demos:

- command schemas;
- result summaries;
- help/defaults/validation search;
- context bundle format;
- redaction policy;
- structured command proposal format.

This improves the project even if no external AI is enabled.

### Phase B: Minimal Human-In-The-Loop Assistant

Add one or two safe capabilities:

- explain selected result;
- fill a command form from natural language;
- draft a report paragraph from selected result;
- explain an error message.

This can be shown in the first demos, but the demo should not depend on it.

### Phase C: Confidentiality And Audit

Before 0.1:

- project-level AI policy;
- preview of context to be sent;
- provider/key settings;
- audit log of AI requests;
- local help-only fallback.

### Phase D: Richer Workflow Assistance

After 0.1:

- workflow coach;
- validation assistant;
- teaching assistant;
- notebook/code assistant;
- optional local-model support.

### Phase E: Agentic Workflows

Only after the command/session engine, trust model, project replay, undo and validation systems are mature:

- multi-step command planning;
- batch analysis;
- project edits;
- external data retrieval;
- autonomous report building.

Agentic workflows should require explicit permissions, audit, cancellation and review.

## No AI Is A Complete Product

Everything pymetrix does must work with no AI at all: help pages, course notes, forms and
generated Python. This is a guarantee, not a setting. Any assistant ships as an optional extra
or plugin, so an institution that does not install it has the whole product, and the `off` mode
above covers installations where it is present. The demos never depend on AI.

## Teaching Modes

Teaching needs its own limits. Some instructors will not want an assistant doing the learning,
and some institutions bar AI from assessments. When an assistant exists, it should respect a
mode set by the instructor:

1. **No AI:** help, notes, source papers, forms and generated Python only. The default for
   course material.
2. **Hint:** conceptual hints; no answers and no code.
3. **Tutor:** explains results and steps.
4. **Instructor:** may draft solutions and report text, for the instructor's own use.

Course material would declare its mode in a note's front matter, for example `ai: hint`, beside
the `links:` it already has, and the assistant would honour the strictest mode among the notes a
student has open. None of this is built: there is no assistant yet. Until there is, every course
is in no-AI mode.

## Recommendation

pymetrix should integrate AI assistance earlier rather than later, but should start with a transparent, human-approved assistant over structured objects.

The right early product story is:

> pymetrix is a reproducible Python macroeconometrics workbench with guided forms, generated code and an assistant that understands the workflow without taking control away from the user.

The wrong early product story is:

> pymetrix is an autonomous AI economist.

The first is credible, useful and aligned with successful open-source AI-assisted workbench patterns. The second would raise trust, confidentiality and validation problems before the core project is ready for them.
