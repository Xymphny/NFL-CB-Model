---
status: proposed          # proposed | accepted | rejected | superseded
date: YYYY-MM-DD
supersedes: null
superseded_by: null
ledger_id: null           # the migration/ledger.yaml row this decides, if any
---

# NNNN. <short imperative title>

## Context and Problem Statement

What was true that forced a decision. Enough that someone who was not here can
tell whether the situation still holds.

## Considered Options

- Option A
- Option B

## Decision Outcome

Chosen: <option>, because <reason>.

## Consequences

What gets better, what gets worse, what is now harder to undo.

## Recovery

THE LOAD-BEARING SECTION FOR ANYTHING DROPPED OR DEFERRED. A standard ADR
records a decision; this project also needs to be able to walk it back, so a
drop record must say exactly where the thing went:

- **Git tag / commit**: `archive/<ledger-id>` or a full SHA
- **Paths at that commit**: the files that made up the component
- **Data artifacts and where they now live**: every path, and whether it is
  raw / training / derived / cache per the ledger's taxonomy
- **What it would take to bring it back**: honestly, in hours or days
- **What has changed since that would break it**: interfaces it assumed,
  schemas it read

An ADR that says "we dropped X because it was not worth it" and stops is the
silent-abandonment failure with extra steps.

## Revisit Triggers

What specific, observable thing would make this decision wrong. Not "if we
have more time" -- something you would actually notice:

- e.g. "if the attempt log ever shows RMS t above 1.5, the shrinkage weight is
  meaningful and this should be reopened"
