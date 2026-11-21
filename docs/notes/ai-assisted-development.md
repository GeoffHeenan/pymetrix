# AI-Assisted Development And Contribution Policy Notes

Draft note for the pymetrix repository, October 2026.

> The policy itself is in [GOVERNANCE.md](../../GOVERNANCE.md#ai-assisted-contributions) and
> [CONTRIBUTING.md](../../CONTRIBUTING.md#ai-assisted-contributions); this note records the
> reasoning and the evidence behind it. Its suggested texts below have been adopted in those files.

## Why This Note Exists

pymetrix has been developed with substantial AI assistance. That should be visible, governed and testable rather than hidden.

The project should not treat AI-assisted development as automatically suspect, nor as automatically acceptable. Human maintainers remain responsible for design choices, accepted code, validation, licensing compliance and releases.

## Evidence From Recent Open-Source Research

A recent empirical study, "Making AI Visible, Not Vanished: How AI Policies Reshape Developer Experience on GitHub", analyzed 29,624 GitHub repositories and identified 385 projects that had adopted AI policies ([arXiv:2608.03329](https://arxiv.org/abs/2608.03329)).

The paper derives a five-part framework called TRACE:

- Transparency: whether AI use is disclosed.
- Responsibility: whether humans remain accountable for the contribution.
- Attribution: whether AI-generated or AI-assisted material is identified.
- Constraints: what uses of AI are limited or forbidden.
- Enforcement: how maintainers check and enforce the policy.

The paper's main practical conclusion is that open-source AI governance usually regulates AI-assisted development rather than banning it. Policies emphasizing transparency and responsibility were associated with stronger community and quality outcomes than purely restrictive approaches.

That finding fits pymetrix's needs. The project should make AI use visible, keep humans accountable, and require validation, without pretending AI-assisted work can or should be excluded.

## Recommended pymetrix Position

AI-assisted contributions are accepted on the same terms as any other contribution.

The contributor is responsible for the work and must be able to understand it, explain it, test it and certify it under the project's contribution rules.

AI tools must not be used to reproduce proprietary code, confidential material, non-public examples, restricted institutional material, or material from commercial tools that the contributor is not free to share.

Substantial AI assistance should be disclosed in the pull request.

Maintainers may ask for tests, derivations, references, cross-checks or independent validation for any contribution, whether or not AI tools were used.

## Applying TRACE To pymetrix

### Transparency

Contributors should disclose substantial AI assistance in pull requests. A short note is enough:

> AI assistance: used for initial implementation draft and test scaffolding; contributor reviewed and revised the final code.

Minor AI use, such as spelling edits or local refactoring suggestions, does not need detailed disclosure unless it materially shaped the contribution.

### Responsibility

The contributor, not the AI tool, is responsible for the contribution.

The contributor must be able to:

- explain the design;
- justify the statistical or numerical method;
- identify the sources used;
- run the relevant tests;
- certify the contribution under the Developer Certificate of Origin.

### Attribution

The project does not need to list AI tools as authors. Authorship and credit belong to human contributors.

For large AI-assisted contributions, the pull request should record which parts were AI-assisted at a high level, such as implementation draft, documentation draft, test data generation, or code review.

### Constraints

AI tools must not be used to:

- copy or approximate proprietary source code;
- reconstruct confidential examples or institutional training material;
- translate non-public code from EViews, MATLAB toolboxes, IRIS, OGP projects, Stata, commercial packages or employer repositories;
- bypass clean-room rules;
- generate code the contributor does not understand;
- add methods without references, tests and validation.

For numerical and econometric code, AI assistance does not reduce the standard of evidence. Native implementations still need independent checks where possible.

### Enforcement

Maintainers may request:

- disclosure of substantial AI use;
- additional tests;
- references to public documentation or papers;
- cross-checks against independent software;
- smaller pull requests;
- explanation of generated code;
- removal or rewriting of code with unclear provenance.

A contribution can be rejected if the maintainer cannot establish that it is understood, lawful to submit, and adequately tested.

## Suggested Text For CONTRIBUTING.md

```markdown
## AI-Assisted Contributions

AI-assisted contributions are welcome, provided they meet the same standards as any other contribution.

You are responsible for everything you submit. You must be able to understand it, explain it, test it, and certify it under the Developer Certificate of Origin.

Do not submit AI-generated material that copies, derives from, or attempts to reconstruct proprietary code, confidential material, non-public examples, employer materials, institutional training material, or commercial software internals.

Mention substantial AI assistance in your pull request. A short note is enough, for example:

> AI assistance: used to draft the first implementation and tests; final code reviewed and revised by the contributor.

Maintainers may ask for tests, derivations, references, cross-checks or independent validation for any contribution, whether or not AI tools were used.
```

## Suggested Text For GOVERNANCE.md

```markdown
## AI-Assisted Development

pymetrix has been developed with substantial AI assistance. Human maintainers are responsible for accepted code, design decisions, validation, licence compliance and releases.

AI-assisted contributions are accepted on the same terms as any other contribution. The contributor is responsible for the work, must be able to explain it, and must meet the same testing and validation rules. Substantial AI generation should be mentioned in the pull request.
```

## Practical Rule

The project should not ask "Was AI used?" as the main quality question.

The better questions are:

- Is the contribution lawful to submit?
- Does the contributor understand it?
- Is the method documented?
- Are the tests adequate?
- Is there an independent cross-check where one exists?
- Does the generated code preserve pymetrix's design principles?

That is the right standard for both AI-assisted and human-written work.
