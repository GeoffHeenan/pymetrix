# Contributing to pymetrix

Thank you for helping. This page says how to get set up, what a change needs, and how
contributions are certified. [GOVERNANCE.md](GOVERNANCE.md) says how decisions are made and what
contributors confirm about intellectual property; please read it first.
[docs/adding-features.md](docs/adding-features.md) walks through adding an estimator, test,
solver, data reader or command so that it works everywhere; [docs/roadmap.md](docs/roadmap.md)
says what the project is working towards.
[docs/releasing.md](docs/releasing.md) says how `main` is kept green and how a release is made.

## Ways to help

- **Use it and report what breaks or confuses you.** For the first collaborators, the most
  valuable thing is to run [Lab 3](examples/course/03_forecast_round.ipynb), or your own small
  project, and say where it fails.
- **Validate an area you know:** run the same estimation or test in another package and compare.
  A recorded comparison becomes a test, and you are credited on the
  [validation page](docs/validation.md).
- **Review** a module, an API or the documentation.
- **Write code:** a fix, a missing method, a data connector, an importer, a plugin.
- **Write course material:** lecture notes and notebooks that link to the help pages.

## Setting up

```bash
git clone https://github.com/GeoffHeenan/pymetrix.git
cd pymetrix
pip install -e ".[dev,jupyter]"
pytest
```

`pytest -n auto --dist loadfile` runs the tests in one process per core, as CI does (pytest-xdist,
part of `[dev]`); `--dist loadfile` keeps each test file in one process, so the fits its fixtures
share are made once. On four cores the whole suite takes about 10 minutes instead of 35.

For the desktop app, add `[qt]`. A pull request runs the tests on Linux with Python 3.11 and 3.13;
Windows (Python 3.11) and every Python on Linux run once per release (docs/releasing.md).

## What a change needs

1. **Tests.** New behaviour has tests, and the whole suite passes.
2. **An independent check for every number.** A numerical result, whether an estimator, a test
   or a solver, is checked against something other than pymetrix. In order of preference:
   - the mature library pymetrix wraps, called directly in the test;
   - another package's recorded output (say which package and version);
   - a published table or worked example (cite it);
   - a hand formula;
   - recovery of known true values in a simulation.

   Add the check to the [validation page](docs/validation.md).
3. **Library first.** If a mature, maintained library already does it, wrap it. Write it natively
   only when no library does, or the library's version cannot do what is needed. Say which in
   the pull request.
4. **Documentation.** A help topic or a section of an existing one, with references (see
   [Citations](#citations)); a line in
   the [user guide](docs/guide.md) for anything user-facing; the [known gaps](docs/known-gaps.md)
   updated if you close or find one.
5. **Commands keep their contract.**
   - Dialog commands have a stable ID (`pymetrix:snake_name`), listed in
     `tests/fixtures/command_ids.json`.
   - A new optional field (one with a default) needs nothing more: older projects use the
     default.
   - A field that is renamed, removed or changes meaning needs a higher `schema_version` and a
     migration, so saved projects keep opening.
   - Plugins can check their own commands with `pymetrix.testing`.
   - **Defaults do not change silently.** Scripts and saved steps leave defaults out, so a changed
     default changes what old work computes. `tests/test_defaults_guard.py` fails when one
     changes: for a command, raise its `schema_version` with a migration that writes the old
     default into older steps; for a method, test or filter, say so in CHANGELOG.md; then
     refresh the snapshot (`python tests/test_defaults_guard.py`).
   - **Record what was used.** Where a method chooses something itself (a lag, a bandwidth, a
     smoothing parameter), put the value used in `provenance.resolved`, next to the request.
6. **Style.** Match the code around you: names, comments, docstrings. Defaults that differ from
   other packages are documented on the help page on defaults.

Keep pull requests small and focused; open an issue first for anything large. Until the 0.1
release (see the [roadmap](docs/roadmap.md)), broad code contributions wait until
the contribution terms are settled; reviews, validation cases and course material are welcome
now.

## Citations

A help page cites the work that introduced or defines what it computes: the paper a user should cite
in their own research. Software is acknowledged elsewhere:
- **A wrapped library** is named on the page by its `wraps` line, which links to the library's
  documentation. It is not cited as a reference.
- **A package whose output a test checks against**, such as R's `rjd3tramoseats` or Stata, goes in the
  [validation page](docs/validation.md) with its version, not in the help.
- **The exception:** where an implementation is itself the definition, cite its manual or package. One
  example is X-13ARIMA-SEATS, where the program defines the numbers.

How to write a reference:
- **Form:** `Surname, I. and Surname, I. (Year). Title in sentence case. Journal Volume(Issue), first-last.`
  - Give the journal's full name as printed when the work appeared, e.g. *Journal of the Royal
    Statistical Society, Series B (Methodological)*.
  - A book gives its edition and publisher.
- **Chapters:**
  - **A chapter by its own authors in an edited volume** is a work in its own right. Cite it in full, with
    pages: `Surname, I. (Year). Chapter title. In Editor, I. (ed.), Book Title, volume, first-last. Publisher.`
  - **A chapter of a textbook by the book's authors** is a pointer into the book. Give its number and title,
    and no pages, since pages change between printings: `Book Title, 2nd ed., chapter 4, "Chapter title".
    Publisher.`
    Several chapters each give their title in brackets: `chapters 6 ("Spectral analysis") and 10 ("Covariance-stationary
    vector processes")`.
- **Reuse:** when a work is already cited elsewhere in the help, reuse the identical string, so the same
  work reads the same everywhere.
- **DOIs:** add a DOI, as `doi:10...`, only after checking it resolves to the work: never type one from
  memory. `python scripts/check_references.py` checks every DOI against Crossref, and with
  `--metadata` reports Crossref's year, volume, issue and pages. The `references` workflow runs the same
  check weekly.
- **No DOI:** for a journal article on JSTOR, give its stable ID instead, as `jstor:1827924` (from
  `https://www.jstor.org/stable/1827924`). Try `doi:10.2307/` with the same ID first: many JSTOR items have that DOI.
  Leave books and textbook chapters without one; the chapter number and title find them in any copy.

## AI-Assisted Contributions

AI-assisted contributions are welcome, provided they meet the same standards as any other
contribution.

You are responsible for everything you submit. You must be able to understand it, explain it,
test it, and certify it under the Developer Certificate of Origin.

Do not submit AI-generated material that copies, derives from, or attempts to reconstruct
proprietary code, confidential material, non-public examples, employer materials, institutional
training material, or commercial software internals.

Mention substantial AI assistance in your pull request; the template has a box for it. A short
note is enough, for example:

> AI assistance: used to draft the first implementation and tests; final code reviewed and
> revised by the contributor.

Minor use, such as spelling or small refactoring suggestions, needs no mention. Maintainers may
ask for tests, derivations, references, cross-checks or independent validation for any
contribution, whether or not AI tools were used. The reasoning behind these rules is in the
[AI-assisted development note](docs/notes/ai-assisted-development.md).

## Certifying your contribution (DCO)

pymetrix uses the [Developer Certificate of Origin](https://developercertificate.org/), version
1.1. By signing off a commit, you certify that you wrote it, or otherwise have the right to submit
it under the project's licence, and that you understand the contribution is public. Sign off with:

```bash
git commit -s
```

This adds a line `Signed-off-by: Your Name <your.email@example.org>`, using your real name.

## Conduct

Be courteous and specific. Disagree with ideas, not people. Assume good faith, especially across
the institutional and product loyalties that many of us have. The maintainers may remove content
or contributors that make the project unwelcoming.
