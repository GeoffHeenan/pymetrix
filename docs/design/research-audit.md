# A research lab book: an audit trail that can be verified

Design note, 26 October 2026, revised 27 October · proposed, to be built as a plugin (`pymetrix-labbook`) on hooks
now in core (extension API 1.3) · builds on [commands.md](commands.md) (recorded calls),
[provenance.md](provenance.md) (what a result records) and [assistant-plugin.md](assistant-plugin.md) (an AI over
the MCP tools)

## The question

Can pymetrix keep a research lab book that a reviewer, a journal, a funder or an AI can check? The aim is to reduce
the abuses common in empirical social research.

Most of these abuses are not invented numbers. They are undisclosed choices:

- **Specification searching:** many models are tried and one is reported.
- **Exclusions decided late:** observations are dropped after the result has been seen.
- **Hypotheses after the results** (HARKing).
- **A sample changed until it works:** the sample or outcome is changed until a test passes.
- **Forking paths:** each choice is defensible on its own, but together they are not disclosed.

A lab book is useful if it records the paths not taken, not only the one reported.

## What pymetrix has now

- **Recorded steps.** Almost every dialog builds a recorded API call, the same as its code typed by hand
  ([commands.md](commands.md)). Each `CommandRecord` keeps:
  - the call and its code;
  - when it ran and how long it took;
  - the versions of what it read and wrote;
  - the series it used;
  - whether it succeeded.

  A project replays from its steps.
- **Provenance.** Every result keeps:
  - data fingerprints and vintages, and the sample rule;
  - the rows dropped and why;
  - the library and its arguments;
  - warnings, seeds and convergence.

  See [provenance.md](provenance.md).
- **Typed code is marked.** Typed Python needs trust to replay (`pymetrix.trust`); recorded calls do not.

## What is missing for an audit

1. **The history is a recipe, not a record.** It keeps the path that was kept:
   - Undo (Edit → Undo) removes a step's record;
   - a cancelled step is not recorded;
   - the history can be cleaned up and steps removed.

   That is right for a recipe, but it hides the forks.
2. **Nothing is tamper-evident.** A project file can be edited, or rebuilt cleanly after the fact.
3. **Nothing ties the analysis to a plan** made before the data were seen.
4. **Nothing counts what was tried,** so a report cannot say how many specifications preceded the one shown.

## The proposal, in five layers

### 1. An append-only lab book

There would be a second record beside the history, the lab book, written as each step runs and never edited.
It keeps everything:

- steps that were undone, cancelled, failed, or later removed or cleaned up;
- typed code;
- projects opened and saved, and data read.

Each entry holds:

| Field | |
| --- | --- |
| `seq`, `time` | the entry's number and when it ran |
| `event` | `step`, `undo`, `cancel`, `open`, `save`, `read`, `brief`, `plan`, `note` |
| `call` | the recorded call (or a hash of typed code, with the code) |
| `reads` | the data fingerprints the step read (`data_fingerprint`, vintages) |
| `result` | a summary: the result's type, its fingerprint, key statistics for fits and tests (coefficients of interest, p-values, sample size) |
| `prev` | the hash of the previous entry |
| `hash` | the hash of this entry, `prev` included |

The `prev` and `hash` fields chain the entries together. Changing or removing an entry breaks every later hash, so
editing is detectable even though it cannot be prevented. The history stays as it is, as the clean recipe; the lab
book is the full record. The lab book is saved inside the project (`labbook.jsonl`) and travels with it.

Undo stays. It now adds an `undo` entry instead of erasing what happened.

### 2. The project brief, and the plan, before the data

Two records come before the work, at different levels of detail. Most projects need only the first.

**The project brief: scope and objectives.** The first entry after `start`, written before any data are read. It
gives the rest of the lab book its meaning: without a stated scope, an auditor can count what was tried but cannot
tell whether it was on topic or a fishing expedition. A `ProjectBrief` setting holds, in fields that later work can be
checked against:

- the title and the research question, and why it matters;
- the objectives, and the hypotheses (H1, H2, ...), each with what would support or contradict it;
- the primary and secondary outcomes;
- the intended data: sources, vintages, the sample period and units;
- the intended methods, broadly ("a VAR on output, inflation and the policy rate"), not exact specifications;
- the scope's limits: what the project deliberately leaves out;
- the team and their roles, the funder, and any ethics or data-access conditions.

Recording it writes a `brief` entry. In the app, a *Project brief* dialog opens when a lab-book project starts. It can
be skipped, and skipping is itself recorded.

- **Amendments, never edits.** A change of scope writes a new `brief` entry with its reason, chained like any other,
  so the record shows how the project evolved and when. That is honest, not damning: projects change. What matters
  is that the change is visible and dated.
- **Steps linked to objectives.** A step can be tagged with the objective or hypothesis it addresses, in its dialog
  or in code (`with objective("H2"): ...`). `verify` then reports:
  - objectives with no analysis: dropped, or quietly abandoned;
  - analyses linked to no objective: exploratory, which is normal, but labelled as such;
  - outcomes and hypotheses added after results on them were seen.
- **Notes.** The same mechanism records free text: the reasoning a paper lab book holds ("dropped the 2020 quarters:
  the survey changed its method"). A *Note* dialog and `labbook.note(...)` write `note` entries, linked to the step
  or objective they concern.
- **Private until published, with its date proved (commit-reveal).** A researcher may not want to disclose
  hypotheses before publishing, for fear of being scooped or copied. The lab book can then record only the brief's
  hash. That hash is time-stamped outside pymetrix (OSF, an RFC 3161 time-stamping authority, or a journal's
  registered-report system), and the full brief is revealed later. Anyone can check that the revealed brief has the
  registered hash. This proves when the brief was written, and that it was written before the work, without making
  the ideas public early.

**The pre-registered analysis plan: exact specifications.** Optional and stricter, for confirmatory work. A researcher
can lock a plan before the data are read. A `Plan` setting holds:
- the hypotheses;
- the primary outcomes;
- the model specifications, as recorded calls;
- the sample and exclusion rules;
- the data request, a `DataRequest` as in [plugins.md](plugins.md).

Locking it writes a `plan` entry whose hash covers the plan. The lab book can then show that the first `read` of
those data came after it.

Each later step is labelled either as **following the plan** (its call matches a planned one) or as **exploratory**.
A deviation report lists the departures, for example: "sample rule changed at entry 214, after 6 fits of the primary
outcome; plan named 3 controls, the reported model has 5." Exploratory work is normal. The point is that it is
labelled.

### 3. Counting what was tried

From the lab book, pymetrix can say:
- how many specifications of each outcome were fitted;
- how many tests were run, and which exclusions and samples were tried.

An exported report can carry this automatically, for example "the model shown is 1 of 23 fitted for this outcome",
with a specification curve drawn from the lab book. Forking paths are then disclosed by default, not by the
researcher's memory.

### 4. Verification

`pymetrix verify project.pymx` runs deterministic checks, with no AI involved:

1. **The chain.** Every hash is recomputed, and any break is shown with the entry where it occurs.
2. **Replay.** The recorded steps are run again from the data. Fingerprints must match, and so must each result's
   fingerprint and key statistics.
3. **The report.** Every number in an exported report is traced to a lab-book entry and comes out the same on replay.
4. **What replay cannot check.** It lists typed Python, data read from outside without a fingerprint the verifier can
   reproduce, and steps with random draws but no seed.

The output is a verification report, readable by a person and by the auditing tools below.

### 5. An AI auditor, on top of the checks

Read-only MCP tools extend the existing server (`pymetrix.mcp_server`). They provide:
- `labbook_summary`;
- `labbook_entries` (by kind, by outcome, by time);
- `deviations`;
- `specification_count`;
- `verify_report`.

An assistant uses them to answer an auditor's questions, citing entries:
- Were the exclusions decided before the outcome was first fitted?
- Does the paper's main model match the plan?
- Did the sample change right after a non-significant result?
- How many outcomes were tried before the reported one?

The auditor flags patterns for a person to judge. It never gives a verdict on its own, and it starts from the
deterministic verification, not from the raw record. The confidentiality modes of
[assistant-plugin.md](assistant-plugin.md) apply: by default it sees the lab book's structure, fingerprints and
summaries, not data values.

## Built as a plugin: `pymetrix-labbook`

The lab book is a plugin, installed separately, rather than part of pymetrix:

- **Opt-in, and a plugin institutions can require.** A journal, funder, course or data custodian can ask for it,
  and an institution can install it by default.
- **Independent of the work it audits.** The auditor (layer 5) can be a separate plugin again, run by the reviewer
  rather than the researcher.
- **Its own dependencies.** Signing and time-stamping stay out of pymetrix's core.
- **Its own pace.** It can change without a pymetrix release.

### What core provides (built: extension API 1.3)

These hooks are small and useful beyond the lab book (activity logs, teaching analytics):

1. **Session events** (`pymetrix.commands.events`). A plugin registers a `SessionObserver` in the `observers`
   group, and every session tells it what happens: in scripts, notebooks, the app and the MCP server alike. The
   events (`SessionEvent.kind`):
   - `step` and `failed`: with their records;
   - `cancelled`: with the step's code, though it leaves no record in the history;
   - `undone`: with the records removed;
   - `removed`: for example, dead ends cleaned up;
   - `opened` and `saving`: a project opened or about to be written.

   Each event says whether it happened during `work`, `recalculation` or `opening`. An observer that fails is
   reported once and skipped: it can never stop the work.
2. **Project attachments.** `session.attachments[key]` holds a plugin's files (name → bytes). They are saved in the
   project under `attachments/<key>/`, reopened with it, and kept as they are when the plugin is missing, so the
   lab book is never lost by opening a project without it.
3. **The save log.** The manifest's `save_log` lists every save: when, by which pymetrix, and which recording
   observers were active. Work done without the lab book therefore shows as a gap, for example "saved 3 times
   since entry 214 with no recorder active", rather than passing silently.

### What the plugin does

- **Writes the chain.** An observer appends a hash-chained entry for each event (layer 1). It keeps the chain in
  `session.attachments["labbook"]`, updated at `saving`.
- **Plans** (layer 2): a `Plan` setting and its dialog, through the extension API's recorded calls and dialogs.
- **Counts and reports** (layer 3): specification counts and a verification appendix, as report templates.
- **Verifies** (layer 4): `pymetrix-labbook verify project.pymx`, a command-line tool that checks the chain, replays
  the steps with pymetrix's public API, and reads the save log for gaps.
- **Shows it in the app:** a Lab book panel and workspace.
- **Later, the auditor's tools** (layer 5). That needs one more hook: a way for plugins to add read-only tools to the
  MCP server.

### What a plugin cannot do

It records nothing before it is installed. "On by default for every project" therefore becomes "installed by
default where it matters", and the save log makes the absence visible. A lab book started on an existing project
says so in its first entry, which records the history it found.

## The hard limit, and how to narrow it

No software stops a researcher from analysing the data elsewhere first, then repeating only the final model in
pymetrix to produce a clean lab book. The lab book proves what happened inside it, not that nothing happened
outside it. There are two ways to narrow the gap:

- **External time-stamping.** The plan entry, and the chain's latest hash at intervals, are anchored outside pymetrix:
  an RFC 3161 time-stamping authority, a commit to a public repository or OSF, or a journal's registered-report
  system. Then even the researcher cannot backdate the plan.
- **Signed data releases.** This is the strongest option. Where data are released under control, by a trusted
  research environment, a statistical office, or a store such as PSETS, the custodian signs each release with its
  fingerprint and time. The custodian releases only after a plan is registered. The lab book's first `read` then
  becomes provably the first access. PSETS already records the publisher's release as each series' vintage
  ([plugins.md](plugins.md)), and signing its plug-in API's releases would be the natural extension.

## Design questions

1. **Privacy.** The lab book stores fingerprints, metadata and result summaries, not data values, so it can go to an
   auditor or journal without the confidential data. Restricted projects may also need entries the auditor sees only
   as hashes, with a process to reveal them.
2. **Typed Python.** Typed code breaks replay. There are two options:
   - an "audited" project mode that refuses typed code;
   - recording each typed step with the hash of its code, marked "trusted, not replayed".

   *Recommended:* the second, with the verification report counting such steps, so audited work is not forced
   into dialogs alone.
3. **Signing.** Hash-chaining makes edits evident but does not say who wrote the entries. Signing entries, or the
   chain's head at each save, with the researcher's key adds that. It needs the credentials helper discussed in
   [plugins.md](plugins.md) (decision 2).
4. **Size.** Exploration produces many entries. Result summaries keep only what an auditor needs. Full results stay
   recomputable by replay, not stored.
5. **Incentives.** If the lab book feels like surveillance, people will explore elsewhere. Exploration should be free
   and normal: counted and disclosed, not penalised. Its value comes from being accepted by journals, funders and
   data custodians, as registered reports are.
6. **Teaching.** The same record shows a student's path through an exercise, which may suit a course better than a
   final answer. Courses would decide whether it is shared.

## Stages

1. **The core hooks** (session events, attachments, the save log): built, extension API 1.3.
2. **The plugin's lab book and `verify`** (layers 1 and 4): the chain written from the events, and the checks of
   the chain, the replay and the save log. These are useful on their own for reproducibility. *Built, as a first
   version:* `packages/pymetrix-labbook`, with `pymetrix-labbook verify` and `show`. Each entry records who acted,
   a person or an AI agent over MCP (extension API 1.5, `SessionEvent.actor`).
3. **The project brief, notes and plan registration** (layer 2). *Started:* the brief and its amendments, notes,
   `objective(...)` and `verify`'s report are built, in Python and in the app's Lab book panel, and so is the analysis plan
   with its deviation report, and commit-reveal for a private brief: stage 3 is built.
   - the brief and its amendments, and notes;
   - steps linked to objectives, and `verify`'s report of unaddressed objectives and unlinked analyses;
   - commit-reveal for a private brief;
   - then the stricter analysis plan and its deviation report.
4. **Counting what was tried, and specification curves in reports** (layer 3). *Built:* each fit's outcome, sample and
   estimates in the lab book; `tried`, `specification_curve` and the counts in `verify` and the panel. Exported
   reports carry the line when asked (Export report's Plugins' notes, through the report-notes hook, extension API
   1.6). Stage 4 is built.
5. **The read-only MCP audit tools and an AI auditor** (layer 5), with an MCP extension hook in core. *Built:*
   `pymetrix-labbook audit project.pymx`, a separate read-only MCP server over the project file (summary, entries,
   brief, deviations, specifications, verify report), so no core hook was needed; decision 5 below taken as
   recommended.
6. **External time-stamping and signed data releases,** designed with PSETS and data custodians.

## Decisions needed

1. **Is the lab book installed by default?** As a plugin, it cannot be on for every project. *Decided (November
   2026):* not yet. It stays a separate add-in, installed on purpose (`pip install pymetrix[labbook]`), and is not in
   `[all]` while it is at this stage. The save log shows where it was not installed.
2. **Typed code in audited projects:** refused, or recorded as trusted but not replayed. *Recommended:* recorded and
   counted (design question 2).
3. **The plan format:** pymetrix's own `Plan`, or a registry's format (OSF, AEA RCT Registry) imported and exported.
   *Recommended:* its own, with export to the registries.
4. **The brief's fields:** pymetrix's own, or an existing standard such as the OSF preregistration templates or
   the AEA RCT Registry's fields. *Recommended:* pymetrix's own short form, with optional sections that map onto
   those templates, so a brief can be exported to a registry when a project registers there.
5. **Where the AI auditor lives:** in the assistant plugin, or a separate audit plugin that a journal or funder can
   run without the researcher's assistant. *Recommended:* separate, so the auditor is independent of the audited.
