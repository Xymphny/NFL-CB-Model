"""CLV in probability when the LINE moved, not just the price.

THE MEASUREMENT THIS REPLACES
`pricing.closing_line_value` reports prob_points as the close's fair
probability minus the bet's implied probability. The close's fair probability
is for the CLOSE's line. When the line moved -- which on a spread is how the
market moves, at an unchanged -110 -- those are two different bets, and the
figure collapses to minus the vig whatever the line did. A paper trade that
beat every close by a full point read -2.4 probability points.

WHAT THIS DOES INSTEAD
The devigged close says: at line Lc, the selection wins with probability pc
(pushes excluded, as a two-way price excludes them). Find the margin mean at
which the league's own margin distribution gives exactly that, then ask the
same distribution what the BET's line was worth. The league's shape carries
the key numbers, so a move from +3 to +3.5 is worth what P(margin = 3) says
and not a flat rate per point:

    NFL, CFB  discrete, sd = the model's MARGIN_SD, measured key-number weights
    NBA       continuous, sd = the fitted sigma_constant

Only the SHAPE comes from the model, never its mean: mu is solved from the
close, so the result is the market's own number moved along the market's
own implied distribution. Leagues without a margin family here (MLB run
line, NHL puck line: both fixed at 1.5) and unmoved lines return None, and
callers keep the price-only figure, which is correct when the line did not
move.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
_K = np.arange(-120, 121)


@lru_cache(maxsize=None)
def family(league: str) -> tuple[float, bool, dict[int, float] | None] | None:
    """(sd, discrete, home-margin key weights) for `league`, or None."""
    if league == "nfl":
        from coverline.leagues.nfl import model as m
        return m.MARGIN_SD, True, m._load_key_number_weights()
    if league == "cfb":
        from coverline.leagues.cfb import model as m
        w = m.KEY_NUMBER_WEIGHTS
        return m.MARGIN_SD, True, (dict(w) if w is not None else None)
    if league == "nba":
        art = json.loads((ROOT / "data" / "nba_fitted.json").read_text())
        return float(art["sigma_constant"]), False, None
    return None


def _cover_nopush(mu: float, line: float, sd: float, discrete: bool,
                  w: np.ndarray | None) -> float:
    """P(win | not push) for a selection at `line` whose margin has mean mu."""
    t = -line                                     # wins when margin > t
    if not discrete:
        return float(stats.norm.sf(t, mu, sd))
    mass = stats.norm.cdf((_K + 0.5 - mu) / sd) - stats.norm.cdf((_K - 0.5 - mu) / sd)
    if w is not None:
        mass = mass * w
    win = mass[_K > t].sum()
    lose = mass[_K < t].sum()
    return float(win / (win + lose))


def fair_at_bet_line(*, league: str, side: str | None, fair_close: float,
                     close_line: float | None, bet_line: float | None) -> float | None:
    """The close's fair probability, moved to the bet's line. None when the
    lines agree, a line is missing, or the league has no margin family."""
    if close_line is None or bet_line is None or float(close_line) == float(bet_line):
        return None
    fam = family(league)
    if fam is None or side not in ("home", "away") or not 0.0 < fair_close < 1.0:
        return None
    sd, discrete, weights = fam
    w = None
    if discrete and weights:
        # Weights are for the HOME margin; the away selection's margin is its
        # negation, so its weight at k is the home weight at -k.
        sign = 1 if side == "home" else -1
        w = np.array([weights.get(int(sign * k), 1.0) for k in _K])

    lo, hi = -80.0, 80.0                          # P(cover) rises with mu
    for _ in range(60):
        mid = (lo + hi) / 2
        if _cover_nopush(mid, close_line, sd, discrete, w) < fair_close:
            lo = mid
        else:
            hi = mid
    return _cover_nopush((lo + hi) / 2, bet_line, sd, discrete, w)
