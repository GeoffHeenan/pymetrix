## What this changes

<!-- One or two sentences. Link the issue if there is one. -->

## How it is checked

<!-- Tests added or changed. For any number pymetrix computes: what independent reference it is
checked against (the wrapped library, another package's output with its version, a published
table, a hand formula, or recovery in a simulation). See CONTRIBUTING.md. -->

## Checklist

- [ ] The test suite passes (`pytest`).
- [ ] New numerical results have an independent check, added to `docs/validation.md`.
- [ ] Documentation updated: help topic, `docs/guide.md`, or `docs/known-gaps.md` as needed.
- [ ] Commands keep their contract (stable ID, schema version and migration if fields changed).
- [ ] Every commit is signed off (`git commit -s`) under the Developer Certificate of Origin.
- [ ] No proprietary, confidential or restricted institutional material (see GOVERNANCE.md).
- [ ] AI tools generated a substantial part of this change. If ticked, say which parts, for example:
      "AI assistance: used to draft the first implementation and tests; final code reviewed and revised by me." 
