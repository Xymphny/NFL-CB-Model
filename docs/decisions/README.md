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
| [0003](0003-attempt-log-and-the-derived-shrinkage-weight.md) | accepted | Reconstruct the attempt log, and treat the weight it yields as a ceiling |
| [0004](0004-ngs-team-code-mismatch.md) | accepted | Record the NGS team-code mismatch; do not fix it in the same breath |
| [0005](0005-nhl-and-nba-structure-without-coefficients.md) | superseded | Ship NHL and NBA as structure, with no fitted coefficients |
| [0006](0006-nhl-and-nba-fitted-walk-forward.md) | accepted | Fit NHL and NBA within season, and ship the graded parameters |
| [0007](0007-the-nhl-puck-line-is-a-rules-problem.md) | accepted | The NHL joint distribution is a rules problem, not a correlation |
| [0008](0008-ship-the-nhl-rules-layer-and-the-puck-line.md) | accepted | Ship the NHL rules layer, and release the puck line |
| [0009](0009-grade-every-league-on-calibration.md) | accepted | Grade every league on calibration, not only on beating a book |
| [0010](0010-refresh-the-nhl-pull-table-and-check-shape.md) | accepted | Refresh the NHL pull table, and check shape rather than only likelihood |
| [0011](0011-a-withheld-market-must-be-refused-by-the-object.md) | accepted | A withheld market must be refused by the object, not only by the list |
| [0012](0012-the-nba-sigma-claim-was-bucketing-noise.md) | accepted | The NBA sigma claim was bucketing noise |
| [0013](0013-the-missing-grid-reconstructed.md) | accepted | Reconstruct the missing grid without selecting anything |
| [0014](0014-a-ceiling-on-what-counts-as-an-attempt.md) | accepted | Put a ceiling on what counts as an attempt |
| [0015](0015-the-away-side-and-the-total-were-both-mispriced.md) | accepted | The away side and the total were both mispriced |
| [0016](0016-clv-is-averaged-in-probability-not-cents.md) | accepted | Average CLV in probability points, never in American cents |
| [0017](0017-mlb-has-the-same-two-rule-problem-as-hockey.md) | accepted | MLB has hockey's two-rule problem, and it lands on the other market |
| [0018](0018-ship-the-mlb-ninth-inning-layer.md) | accepted | Ship the MLB ninth-inning layer, and reopen the moneyline on the gate that closed it |
| [0019](0019-the-cfb-cache-carries-impossible-results.md) | accepted | The CFB cache carries results its league forbids |
| [0020](0020-audit-every-frame-for-impossible-results.md) | accepted | Audit every frame for impossible results, on purpose |

Start a new one from [0000-template.md](0000-template.md). The **Recovery** and
**Revisit Triggers** sections are not optional for anything dropped or
deferred; they are the whole reason the template is not just MADR.
