---
status: accepted
date: 2026-09-22
supersedes: null
superseded_by: null
ledger_id: league-calibration
---

# 0023. NFL dispersion moves, and cannot be forecast from its own history

## Context and Problem Statement

ADR 0009 found that NFL residual dispersion varies across seasons — 11.73 in
2022 to 15.15 in 2021, with Bartlett rejecting equal variance at p = 0.014 —
and deliberately did not change `MARGIN_SD`:

> Knowing the value moves is not the same as being able to forecast it. A
> season-varying sd has to predict next season's dispersion and be graded on
> that, and no such forecast exists.

That left the open item honest but unexamined: the constant was kept because
nobody had done the work, not because the work had been done.

## Decision Drivers

- `MARGIN_SD` is what every NFL probability divides by and therefore what every
  Kelly fraction divides by.
- An open item that reads "we haven't checked" should eventually read "we
  checked."

## Considered Options

- **Leave it open.** Costs nothing and settles nothing.
- **Build the forecast and grade it, prepared to fail.** Chosen.

## Decision Outcome

**No forecast beats the constant, and the most obvious one is significantly
worse.** Train 2014-2019 to seed the estimators, graded **once** on 2020-2023
(831 games), four candidates fixed a priori with zero tuned parameters:

| candidate | gain vs constant | t |
|---|---|---|
| last season's realised sd | −0.0132 | **−2.65** |
| shrunk halfway to the prior mean | −0.0045 | −1.64 |
| expanding mean of prior seasons | −0.00002 | −0.01 |

The expanding mean landing on −0.01 is the sanity check that the comparison is
sound: it and the shipped constant are both the long-run mean, so they should
be indistinguishable, and they are.

**Variance that moves is not variance that is predictable**, and the
decomposition says why. The season sds vary with a standard deviation of
**1.048**; sampling alone on ~195 games a season accounts for **0.695** of
that, leaving a real signal near **0.78**. So Bartlett was right — the
variation is real. But the **lag-1 autocorrelation of the season sd is −0.32 on
nine pairs, against a standard error of 0.33**: negative, and
indistinguishable from zero. A quantity whose own history carries no sign of
itself cannot be forecast from that history.

`MARGIN_SD` stays at 13.2979, now **for a measured reason** rather than an
unexamined one, which was the whole point of asking.

## Consequences

**The rejection is logged.** `evidence/attempts.yaml` gains
`nfl-season-varying-dispersion` at t = −2.65, and the shrinkage weight moves
from 0.9327 to 0.9301 pooled and 0.9168 to 0.9141 robust. A decisive **no**
raises `E[t²]` exactly as much as a decisive yes, because t is squared, and a
log that kept only successes would return a weight that ships noise at full
size.

**Two gentler variants are recorded in the artifact and not logged separately**,
because they are the same question asked three ways. Logging each would
triple-count one attempt in a weight that is quadratic in t.

**This does not say the dispersion is constant.** It says it is unforecastable
from its own history. A predictor built from something *else* — pace, weather,
rule changes, a scoring-environment index — is a different question and is
untouched here.

## Recovery

`model/forecast_nfl_dispersion.py` re-runs the whole grade from the committed
cache. Nothing shipped changed, so there is nothing to revert.

## Revisit Triggers

- **Seasons accumulate.** Nine autocorrelation pairs is few; the standard error
  is 0.33 and the estimate is −0.32, so the honest reading is "no evidence",
  not "proof of none."
- **A predictor outside the sd's own history is proposed.** That is a different
  model and would need its own grade; this record rules out only the
  autoregressive family.
- **The realised sd leaves the 11.7-15.2 range.** A genuine regime change would
  make the long-run mean a worse anchor, and the constant's defence rests on
  that mean.
- **CFB's constant stops holding up.** It was checked the same way out of
  sample and does; the two leagues are being treated differently on evidence,
  not on principle.
