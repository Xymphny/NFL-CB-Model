"""How much of the model to trust OVER THE MARKET, measured against the market.

WHY THIS EXISTS
Staking blends the model toward the market before sizing:

    p_used = w * p_model + (1 - w) * p_market      (core.staking.shrink_probability)

Until now w came from evidence/attempts.yaml, whose every attempt is graded
MODEL AGAINST MODEL -- does a change beat the version before it. That answers
"is the model getting better", not "does the model know anything the price
does not". A model can improve steadily and still add nothing to a closing
line, and then any w above zero manufactures an edge out of disagreement
alone.

WHAT IS ESTIMATED
The w that maximises the likelihood of what actually happened, using the
exact blend staking applies:

    L(w) = sum  y log q + (1 - y) log(1 - q),   q = w p_model + (1 - w) p_market

over settled bets the model priced before they were played. L is concave in
w, so the maximum is unique. w = 0 means the market alone is best; w = 1
means the model alone is; w < 0 means leaning AWAY from the model beats the
market, i.e. the model's disagreements point the wrong way.

WHAT IS SHIPPED IS A LOWER BOUND, NOT THE ESTIMATE
Staking with the point estimate stakes on a coin flip whenever the estimate
is noise. The staking weight is the one-sided lower confidence bound
w_hat - z * SE, clipped to [0, 1]: zero unless the data rule zero out. A
bound, not a test, because the quantity sizes money continuously -- a
significant but tiny w should stake tiny amounts, not full ones.

SE is from the observed information at the maximum. Where the maximum sits on
the edge of what keeps every q inside (0, 1), the estimate is still reported
and the bound is still used, but the result says so.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

#: One-sided 95%. Fixed here rather than tuned: there is one estimate per
#: league and nothing to search over.
Z_ONE_SIDED = 1.6449

_EPS = 1e-9


@dataclass(frozen=True)
class MarketWeight:
    n: int
    w_hat: float
    se: float
    staking_weight: float
    loglik_gain_per_bet: float   # at w_hat, over the market alone
    gain_t: float                # paired t of per-bet gain at w_hat
    mean_disagreement: float     # mean |p_model - p_market|
    at_boundary: bool

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


def _feasible(pm: np.ndarray, pk: np.ndarray) -> tuple[float, float]:
    """The range of w for which every blended q stays strictly inside (0,1)."""
    d = pm - pk
    lo, hi = -np.inf, np.inf
    for bound in (_EPS, 1 - _EPS):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (bound - pk) / d
        pos, neg = d > 0, d < 0
        if bound == _EPS:
            lo = max(lo, np.max(r[pos], initial=-np.inf))
            hi = min(hi, np.min(r[neg], initial=np.inf))
        else:
            hi = min(hi, np.min(r[pos], initial=np.inf))
            lo = max(lo, np.max(r[neg], initial=-np.inf))
    return float(lo), float(hi)


def estimate(p_model: Sequence[float], p_market: Sequence[float],
             outcome: Sequence[int], z: float = Z_ONE_SIDED,
             search: tuple[float, float] = (-3.0, 4.0)) -> MarketWeight:
    """Maximum-likelihood blend weight, its SE, and the staking bound."""
    pm = np.asarray(p_model, float)
    pk = np.asarray(p_market, float)
    y = np.asarray(outcome, float)
    if not (len(pm) == len(pk) == len(y)) or len(y) < 30:
        raise ValueError(f"need at least 30 aligned rows, got {len(y)}")
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError("outcomes must be 0 or 1; drop pushes before calling")
    for name, p in (("p_model", pm), ("p_market", pk)):
        if np.any((p <= 0) | (p >= 1)) or np.any(~np.isfinite(p)):
            raise ValueError(f"{name} must lie strictly inside (0, 1)")

    d = pm - pk
    flo, fhi = _feasible(pm, pk)
    lo, hi = max(search[0], flo), min(search[1], fhi)

    def score(w):          # dL/dw
        q = pk + w * d
        return float(np.sum(d * (y / q - (1 - y) / (1 - q))))

    # Concave => the score is decreasing; bisect for its zero.
    if score(lo) <= 0:
        w, boundary = lo, True
    elif score(hi) >= 0:
        w, boundary = hi, True
    else:
        a, b = lo, hi
        for _ in range(200):
            m = 0.5 * (a + b)
            if score(m) > 0:
                a = m
            else:
                b = m
        w, boundary = 0.5 * (a + b), False

    q = pk + w * d
    info = float(np.sum(d * d * (y / q**2 + (1 - y) / (1 - q) ** 2)))
    se = 1.0 / math.sqrt(info) if info > 0 else math.inf

    per = (y * np.log(q) + (1 - y) * np.log(1 - q)
           - (y * np.log(pk) + (1 - y) * np.log(1 - pk)))
    gain = float(per.mean())
    sd = float(per.std(ddof=1))
    t = gain / (sd / math.sqrt(len(per))) if sd > 0 else 0.0

    bound = min(max(w - z * se, 0.0), 1.0)
    return MarketWeight(n=int(len(y)), w_hat=float(w), se=float(se),
                        staking_weight=float(bound), loglik_gain_per_bet=gain,
                        gain_t=float(t), mean_disagreement=float(np.mean(np.abs(d))),
                        at_boundary=bool(boundary))


# ------------------------------------------------------------ the artifact --


_ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_PATH = _ROOT / "data" / "market_weights.json"


def staking_weight(league: str, path: Path = WEIGHTS_PATH) -> tuple[float, dict | None]:
    """(weight, the league's grade) from data/market_weights.json.

    A league with no row -- or no file at all -- gets 0 and None: no
    market-facing evidence means no stake, never a default that happens to
    be positive. The weight is re-validated here rather than trusted, because
    an artifact edited by hand to 0.9 is exactly the failure this replaces.
    """
    import json
    if not Path(path).exists():
        return 0.0, None
    grade = json.loads(Path(path).read_text()).get("leagues", {}).get(league)
    if grade is None:
        return 0.0, None
    w = float(grade["staking_weight"])
    implied = min(max(float(grade["w_hat"]) - Z_ONE_SIDED * float(grade["se"]), 0.0), 1.0)
    if abs(w - implied) > 1e-4:
        raise ValueError(
            f"{league}: staking_weight {w} is not the lower bound its own "
            f"w_hat and se imply ({implied:.4f}). The artifact was edited or "
            "is from a different estimator; regenerate it with "
            "model/grade_market_weight.py.")
    return w, grade
