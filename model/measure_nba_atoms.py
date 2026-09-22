#!/usr/bin/env python3
"""How much mass sits exactly on an NBA line, and whether a tie is possible.

WHY
NormalMarginDistribution's docstring said NBA margins are "integers too, but
the modelling convention there treats them as continuous because the support
is wide enough that atom mass is small". Nobody had measured "small", and
`discrete=False` makes margin_pmf return zero -- which reads as "no push is
possible" and is false.

WHAT THE MEASUREMENT SAYS
Over 3,540 walk-forward games:

  - An NBA margin lands EXACTLY on the modal spread 3.3% of the time. The NFL
    key-number correction exists because P(margin=3) is about 8%; 3.3% is
    smaller than that and is not zero, which is what the model was asserting.
  - A game NEVER ends level. Overtime resolves every one. The rounded normal
    would put 2.585% on a tie, against an empirical 0.000 -- the same defect
    ADR 0007 found in hockey and ADR 0017 in baseball.
  - Margins of plus or minus one are DEPLETED, at ratios of 0.75 and 0.81.
    Unlike hockey, where the tie-break awards exactly one goal and piles mass
    onto plus and minus one, an NBA overtime is a full five minutes and
    scatters it.

WHAT IS NOT CLAIMED
A key-number table. With about 90 games per margin value the ratios carry a
standard error near 0.10, so the values between 1.0 and 1.2 are noise and only
the tie and the plus-or-minus-one depletion are outside it. Fitting a table to
this would be fitting noise, and NBA has no pristine season to grade one on.

So the action is a REFUSAL, not a correction: integer NBA lines are refused
for want of a measured correction, exactly as CFB's are. Half-point lines are
unaffected, which is most of the board.

Writes model/nba_atoms.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from coverline.leagues.nba.model import load_fitted  # noqa: E402
from model import fit_nba_walkforward as nbawf  # noqa: E402
from model.fit_nba import load  # noqa: E402

OUT = HERE / "nba_atoms.json"


def main() -> int:
    art = load_fitted()
    hp = art["hyperparameters"]
    sd = float(art["sigma_constant"])
    seasons = tuple(art["_provenance"]["tune_seasons"]) + (
        art["_provenance"]["holdout_season"],)
    pred = nbawf.walk_forward(load(seasons), hp["k"], hp["home_adv"],
                              hp["carryover"])
    mu = pred.mu.to_numpy(float)
    act = pred.margin.to_numpy(int)
    n = len(act)

    atoms = {}
    for k in range(-15, 16):
        emp = float((act == k).mean())
        modelled = float(np.mean(stats.norm.cdf(k + 0.5, mu, sd)
                                 - stats.norm.cdf(k - 0.5, mu, sd)))
        se = float(np.sqrt(max(emp, 1e-9) * (1 - emp) / n))
        atoms[str(k)] = {
            "empirical": round(emp, 5),
            "rounded_normal": round(modelled, 5),
            "ratio": round(emp / modelled, 4) if modelled else None,
            "standard_error_of_empirical": round(se, 5),
            "ratio_is_outside_noise": bool(
                modelled and abs(emp - modelled) > 2 * se),
        }

    report = {
        "_provenance": {
            "script": "model/measure_nba_atoms.py",
            "n": int(n), "seasons": [int(s) for s in seasons],
            "sigma": sd,
            "descriptive": True, "graded": False,
            "note": ("properties of the residual distribution; nothing is "
                     "fitted, so no season is spent -- and no NBA season is "
                     "pristine anyway"),
        },
        "ties_are_impossible": {
            "empirical": round(float((act == 0).mean()), 5),
            "rounded_normal_would_say": atoms["0"]["rounded_normal"],
            "why": "overtime resolves every NBA game",
        },
        "modal_atom": {
            "largest_empirical": round(float(max(
                (act == k).mean() for k in range(-15, 16))), 5),
            "compare_nfl_key_number_3": 0.08,
            "why_it_matters": ("discrete=False makes margin_pmf return zero, "
                              "which asserts no push is possible. This is how "
                              "often one happens."),
        },
        "one_point_margins_are_depleted": {
            "plus_one_ratio": atoms["1"]["ratio"],
            "minus_one_ratio": atoms["-1"]["ratio"],
            "why": ("unlike hockey, where the tie-break awards exactly one "
                    "goal, an NBA overtime is five minutes and scatters the "
                    "tie mass instead of piling it onto plus and minus one"),
        },
        "no_table_is_claimed": (
            "about 90 games per margin value gives a ratio standard error "
            "near 0.10, so values between 1.0 and 1.2 are noise. Only the tie "
            "and the plus-or-minus-one depletion are outside it. Fitting a "
            "key-number table to this would be fitting noise, and no NBA "
            "season is pristine to grade one on."
        ),
        "atoms": atoms,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"n={n}  ties: empirical {report['ties_are_impossible']['empirical']} "
          f"vs modelled {report['ties_are_impossible']['rounded_normal_would_say']}")
    print(f"largest empirical atom {report['modal_atom']['largest_empirical']}")
    print(f"+-1 ratios {atoms['1']['ratio']} / {atoms['-1']['ratio']}")
    outside = [k for k, v in atoms.items() if v["ratio_is_outside_noise"]]
    print(f"margins outside sampling noise: {sorted(outside, key=int)}")
    print("wrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
