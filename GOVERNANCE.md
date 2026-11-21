# pymetrix Governance Draft

Draft for discussion with the first collaborators, October 2026. Nothing here is settled until the
first reviewers and collaborators have read it and had a chance to comment.

## Purpose

pymetrix is an integrated workbench for applied macroeconometrics in Python: data, estimation,
testing, model solving, forecasting, reports and reproducible projects in one workflow, usable
from scripts, notebooks and guided forms.

pymetrix is an integration layer, not a new home for every method:

- Where a mature, maintained library implements a method well, such as statsmodels, linearmodels,
  arch, SciPy and others, pymetrix wraps it rather than rewriting it.
- A method written natively must be checked against an independent reference, as described in
  [CONTRIBUTING.md](CONTRIBUTING.md).
- A general-purpose improvement that belongs in an upstream library should go there, with
  pymetrix using it.
- pymetrix aims to work alongside other tools, not to replace them for their users. Importing
  Dynare models, and in future IRIS models and EViews workfiles, is in scope.

## Current Stage

pymetrix is an open-source, founder-led project. Anyone may use it, report problems, review it and
contribute; the first collaborators act as reviewers and advisors. Maintainer status and commit
rights are separate and will be added only after sustained, careful contribution.

pymetrix has been developed with substantial AI assistance. The project lead is responsible for
the accepted code, tests, design decisions and licence compliance of the existing codebase.

## Roles

- **Users** use pymetrix and report problems. Their reports set priorities.
- **Contributors** send changes: code, tests, documentation, course material and validation cases.
- **Reviewers** check work in an area they know, without needing commit rights; for example,
  checking results against another package. Their review is credited.
- **Maintainers** can merge changes and make releases. They are contributors with a record of
  sustained, careful work, invited by the existing maintainers.
- **Project lead:** Geoffrey Heenan, while the project is small. The lead decides when the
  maintainers cannot agree, and is responsible for the project's direction, its name and its
  releases.

## Decision Making

- **Lazy consensus.** Most decisions are made in the open, in issues and pull requests. A proposal
  that draws no objection within a reasonable time, normally a week, is agreed.
- **Objections** come with reasons and, where possible, an alternative. Maintainers try to reach
  agreement; failing that, the project lead decides and records why.
- **Larger decisions** are written up in the design document before they are made. Larger means
  scope, public API, file formats, licence or governance.
- **Moving to a steering group.** When there is a sufficient core of active contributors who can
  move the project forward, at least three active maintainers, they agree a steering group to
  replace the project lead's deciding role as the project lead steps back, and update this
  document.

## Intellectual Property And Clean-Room Rules

pymetrix must be free of others' rights, so that anyone can use it under its licence. Contributors
confirm all of the following:

- **No proprietary material.** Nothing copied or derived from the source code, internal
  documentation or confidential materials of commercial packages, such as EViews, Stata, MATLAB
  toolboxes and others, or of any employer, institution or international organisation, that the
  contributor is not free to share. Published papers, public documentation and published results
  may be used and cited.
- **Clean room for related products.** A contributor who has seen the non-public source code of a
  package that pymetrix overlaps with does not write pymetrix's version of the same feature from
  that knowledge. They are very welcome to review it, test it, and compare its results with that
  package's published output.
- **Employer permission.** Contributors are responsible for having the right to contribute. Work
  done on an employer's time or equipment, or as part of a job, may belong to the employer. Staff of
  institutions with rules on outside activities should check them, and obtain any approval needed,
  before contributing.
- **Data and course material.** Example data and course material must be free to redistribute,
  with their sources and licences recorded. Material from an institution's training programmes is
  not added without its written permission.

## Related Tools And Conflicts

Many people who know this field are tied to related tools: as authors, maintainers, consultants,
trainers or employees. That is an asset, not a barrier.

- **Disclose.** Contributors with such ties mention them to the maintainers, and maintainers list
  theirs in the contributors file.
- **Step back where needed.** On a decision that directly affects a product they are tied to, they
  say so. They may take part in discussion and technical review, but should not be the deciding
  voice on decisions that directly advantage or disadvantage a product they are tied to.
- **Interoperability is welcome.** Making pymetrix read, write or compare with a related tool is
  welcome from anyone, including that tool's authors.

## Contributions, Licence And Credit

- **Licence.** pymetrix is licensed under the BSD 3-Clause License. Contributions are accepted
  under the same licence: inbound equals outbound. There is no contributor licence agreement;
  contributors keep the copyright in their work.
- **Certifying a contribution.** Each commit is signed off under the Developer Certificate of
  Origin, as described in [CONTRIBUTING.md](CONTRIBUTING.md). The sign-off certifies that the
  contributor has the right to submit it under the licence.
- **Credit.** Contributors are listed in a contributors file and in each release's notes.
  Reviewers who validate an area are credited on the [validation page](docs/validation.md).
- **Citation.** A citation file will name the authors to cite. Authorship of papers about pymetrix
  follows substantive contribution, described with the CRediT roles.

## AI-Assisted Contributions

AI-assisted contributions are accepted on the same terms as any other. The contributor is
responsible for the work, must be able to explain it, and must meet the same testing and
validation rules.

Contributors must not submit AI-generated material that they cannot understand, explain, test and
certify under the Developer Certificate of Origin. AI tools must not be used to reproduce
proprietary code, confidential materials, non-public examples or restricted institutional material.

Substantial AI generation should be mentioned in the pull request. Maintainers may ask for tests,
derivations, references or independent validation for any contribution, whether or not AI tools
were used.

How this applies in practice is in [CONTRIBUTING.md](CONTRIBUTING.md#ai-assisted-contributions);
the evidence and reasoning behind it, including the TRACE framework (transparency,
responsibility, attribution, constraints, enforcement), are in the
[AI-assisted development note](docs/notes/ai-assisted-development.md). AI features *inside*
pymetrix are a separate matter: they will propose commands for the user to approve, never act on
their own ([note on AI in the UI](docs/notes/ai-assistance-in-the-ui.md)).

## Commercial Use

The licence allows anyone, including maintainers, to use pymetrix commercially: to teach courses,
consult, or build products on it.

The project name is managed for the benefit of the project. No contributor receives exclusive
commercial rights to it. Maintainers disclose paid work that bears on the project's direction.

## Changes To Governance

Changes to governance are proposed as a pull request and agreed by the maintainers. While there is
a project lead, the lead must also agree.

## Decided

1. **Copyright line.** When others' contributions are first merged, the `LICENSE` copyright line
   becomes "Geoffrey Heenan and the pymetrix contributors".
2. **Steering group.** At least three active maintainers. The decision to move rests on there
   being a sufficient core of active contributors to carry the project forward as the project
   lead steps back (see Decision Making).
3. **Name.** The name is checked and protected on PyPI, and possibly as a trademark, before the
   0.1 release. As of 11 October 2026, `pymetrix` is not taken on PyPI.
4. **Sign-off or agreement.** The DCO sign-off is enough; there is no contributor agreement for
   everyone. Current staff of institutions with outside-activity rules are asked directly whether
   their approval says who owns their contributions. If an employer requires a written agreement,
   it is handled for that contributor only.
5. **Release timing.** The repository is public. After the first collaborators' findings are
   fixed, 0.1 is released on PyPI. The name is
   claimed by that real release, not by an empty placeholder, and its availability is checked again
   before launch. A trademark search, if wanted, is done before launch.
6. **Interoperability, in order.**
   - Dynare `.mod` import, which exists.
   - IRIS model files next.
   - EViews data next: workfiles read through EViews' own Python interface where EViews is
     installed, without reading its file format directly. This is agreed with EViews-connected
     collaborators first.
   - QuantEcon last, shaped by what its maintainers say would help.
7. **AI provenance.** Disclosure is proportionate. The pull-request template asks whether AI tools
   generated a substantial part of the change, and which parts. The same rule applies to code,
   tests, examples and documentation. There is no line-by-line tracking, and no required AI line in
   commit messages. The safeguard is review: maintainers may ask for tests, derivations, references
   or independent validation for any contribution.
