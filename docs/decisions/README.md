# Architecture decision records

One file per decision, numbered, immutable once accepted. A decision that turns
out wrong is SUPERSEDED by a new record, never edited or deleted -- the point is
to be able to see what was believed at the time and why.

`migration/ledger.yaml` is the reverse index: a component dropped or deferred
there names its ADR, and `tests/core/test_migration_ledger.py` fails if that
file is not on disk. So this directory cannot silently fall behind the ledger.

A third index exists deliberately outside these documents: dropped components
are tagged `archive/<ledger-id>` in git, so `git tag -l 'archive/*'` lists them
even if this directory is lost.

| ADR | Status | Decision |
|-----|--------|----------|
| [0001](0001-rebuild-around-a-distribution-seam.md) | accepted | Rebuild around a single distribution seam |
| [0002](0002-odds-data-vendor-and-tier.md) | accepted | Buy The Odds API at the 100K tier, and never split the CLV baseline |

Start a new one from [0000-template.md](0000-template.md). The **Recovery** and
**Revisit Triggers** sections are not optional for anything dropped or
deferred; they are the whole reason the template is not just MADR.
