# pymetrix-labbook

A research lab book for pymetrix, as a plugin: an append-only, hash-chained record of the work, kept inside the
project and checked by replay. It records what a project's history does not keep: steps that were undone,
cancelled, failed or removed, as well as the steps kept. A reviewer can then see the paths not taken, not only the
one reported. The design is in pymetrix's `docs/design/research-audit.md`; this is its first stage.

```bash
pip install -e packages/pymetrix-labbook        # from the repository (not on PyPI yet)
```

Once installed, it records every session (scripts, notebooks, the desktop app and the MCP server) with no set-up.
Each project saved then carries its lab book in `attachments/labbook/labbook.jsonl`.

```bash
pymetrix-labbook verify study.pymx              # check the lab book; exit code 1 if it fails
pymetrix-labbook show study.pymx                # list its entries
```

```python
from pymetrix_labbook import verify

result = verify("study.pymx")    # result.ok, result.errors, result.warnings, result.notes, result.checked
print(result)
```

## The project brief, notes and objectives

Before the work, record what it sets out to do. Steps run inside `objective(...)` are linked to what they address:

```python
from pymetrix_labbook import brief, note, objective

brief(session, "Consumption and income", "How much of a rise in income is consumed?",
      objectives={"O1": "Estimate the marginal propensity to consume"},
      hypotheses={"H1": {"statement": "The MPC is below one", "supports": "a coefficient on y under 1"}},
      primary_outcomes=["the coefficient on y"], data={"sources": "national accounts, 2024 vintage"})
with objective("O1", "H1"):
    session.run(FitEquation("c ~ y", name="fit"))
note(session, "Kept the full sample: no breaks found", about="O1")
```

- **Amendments, never edits:** calling `brief` again restates the brief with a `reason`; both stay in the chain.
- **No brief:** `skip_brief(session, reason)` records that there is none, and why.
- **A private brief:** `commit_brief(session, "brief.json", ...)` writes the brief, with a random salt, to a file you
  keep private, and puts only its hash in the lab book. Register the hash with a time-stamping service (OSF, an RFC
  3161 authority) to prove when it was written. At publication, `reveal_brief(session, "brief.json")` discloses it;
  `verify` checks it matches the commitment and counts it as written when committed. In the app: the brief form's
  "Keep it private" box, and **Reveal…**.
- **In the app:** the Lab book panel (View → Lab book) shows the brief, or a prompt to write one; **Brief…**,
  **Skip…** and **Note…** write the entries; ticking an objective links the steps run while it is ticked; the
  brief's checks and the latest entries update as you work.
- **What `verify` adds:** a brief written after the work began, objectives with no analysis, analyses linked to no
  objective (exploratory, labelled as such), and hypotheses added only after results on them existed.

## The pre-registered analysis plan

For confirmatory work, lock the exact specifications before reading the data:

```python
from pymetrix_labbook import lock_plan

lock_plan(session, [FitEquation("c ~ y", name="fit")], data=["data"], sample="1990Q1-2019Q4", hypotheses=["H1"])
```

Each later step is labelled **planned** (its code matches a specification, whatever its result is called) or
**exploratory**. Changing a locked plan needs a reason. `verify` reports the deviations: planned data read before
the plan was locked, planned specifications not run, exploratory steps since, and each change with the work done
before it. In the app: **Plan…** in the Lab book panel, which offers the session's steps to copy into the plan.

## What was tried

Every fit of an outcome counts, kept, undone or failed: the lab book says how many specifications were tried.

```python
from pymetrix_labbook import specification_curve, tried

tried(session, "fit")    # 'fit is 1 of 23 specifications fitted for c (31 fits, lab book entries 2-58)'
specification_curve(session, "c", "y", highlight="fit").savefig("curve.png")
```

`verify` lists the count for each outcome. In the app, the panel's checks show the counts and the line for the
selected result, and **Specification curve…** draws the specification curve.

In an exported report, tick **Plugins' notes** (Export report, Advanced; `ExportReport(..., plugin_notes=True)`):
a "What was tried" section gives this line for each result. The line comes from the session's record, so the
report's recorded script does not write it again.

## Auditing with an AI assistant

`pymetrix-labbook audit study.pymx` serves read-only tools over the project's lab book to an assistant (Claude
Desktop, VS Code and other MCP clients), for a reviewer, journal or funder:

```json
{"mcpServers": {"audit": {"command": "pymetrix-labbook", "args": ["audit", "C:/review/study.pymx"]}}}
```

The tools: `summary`, `entries` (by kind, outcome or range), `brief`, `deviations` (against the brief and plan),
`specifications` (each fit of an outcome) and `verify_report`. They read the project file and never write it. The
assistant is told to cite entries, to flag patterns for a person to judge, and never to give a verdict. It is
separate from the researcher's own assistant, so the auditor is independent of the audited. Needs
`pip install "pymetrix-labbook[audit]"`.

## What `verify` checks

1. **The chain:** each entry holds the hash of the one before it, so a changed, removed, inserted or reordered
   entry is found, and named.
2. **Coverage:** pymetrix's save log records which recording plugins were active at each save. A save made
   without the lab book, after it began, shows as a gap.
3. **The steps:** the project's steps are those the lab book recorded at its last save, so a project changed
   outside pymetrix, or given another project's lab book, is found.
4. **The replay:** the project is opened, which runs its steps again. Each current result is compared with the
   fingerprint recorded when it was made: estimates to 10 significant digits, and datasets' values exactly.

It lists what it cannot check, rather than passing over it:
- work done before the lab book was installed;
- typed code not run because it is not trusted;
- steps whose plugin is missing;
- results not computed yet.

## What an entry holds

The time, the event and the code of each step, and who acted: the person at the machine (their login, through the
app or Python) or an AI agent (the MCP client, as it names itself, through pymetrix's MCP server). For each step, also:
- the names and versions it read;
- for what it wrote: the type, a fingerprint and the number of observations.

Who acted is hashed with the rest of the entry, so it cannot be changed afterwards. `verify` counts each actor's
steps, and `show` lists them beside each entry.

For a fitted model, also its outcome, sample and estimated coefficients with their standard errors. Entries hold
no data values, so a lab book can go to a reviewer without the data.

## Limits

The lab book proves what happened while it was installed, not that nothing happened elsewhere. Work done outside
pymetrix, or before installing it, leaves no entries. The save log makes gaps visible, but cannot fill them.
External time-stamping and signed data releases, which narrow this, are later stages of the design.

License: BSD-3-Clause.
