"""
Player projection engine -- Stage 1 of the props/bottom-up program.

SOVEREIGN BY DESIGN: consumes only raw historical box-score stats
(nflverse player-week files). It never reads the team betting model's
ratings or predictions, so its output remains eligible as an
independent input to game-line experiments (the circularity
constraint agreed 2026-09-20).

Per player, walk-forward EWMA states (halflife ~4 games) for volume
shares, per-opportunity efficiency, and TD rate, credibility-regressed
to position means; team volume context from raw team stat totals;
opponent adjustment from defenses' allowed-efficiency vs league,
shrunk 50%. States carry across season boundaries with a decay factor
-- players keep their history when rosters change (the property the
sides experiments will lean on later).

DISTRIBUTIONS, not point estimates: P(over line) comes from the
empirical CDF of actual/projected ratios, fit per market on TRAIN
seasons only (<= 2023) and FROZEN; 2024-2025 are held out for the
calibration report. Anytime TD uses Poisson on projected TD rate.
No parametric shape is assumed anywhere -- if the tails are fat, the
ECDF already knows.

Ship rule for consumers: the engine's probabilities appear on prop
cards as a labeled second opinion (watch mode) only if the held-out
calibration report is sane; verdicts require live graded evidence.

V2 (2026-09-20), spec written by the Bijan Robinson discussion:
  1. TD lambda decomposed causally -- player base rate x (current
     team TD environment / the environment his history was earned
     in) x opponent TD-defense multiplier. A QB injury now flows
     into every skill player's scoring odds through the team
     environment EWMA instead of arriving by lag through each
     player's own TD state. Clamped and shrunk like everything else.
  2. Star-tier credibility: the TD shrink target is the (position,
     volume-tier) mean, not a global 0.25, and workhorses (tier 2,
     >= 15 opp/g) get a smaller k than fringe players -- elite
     role-secure players stop being regressed like committee backs.
  3. Per-tier power recalibration P(score) = 1 - exp(-a_tier *
     lam^b) replaces the single pooled lambda multiplier. The b < 1
     concavity is what lets the causal channels in: with raw
     multipliers the high-lambda claims ran ~12pp hot (hot streaks
     mean-revert; a scalar shrink matches means but not the tail),
     and with b fit on train the (env x opp) config beats
     channels-off on train log-loss AND calibrates held-out.
  4. Burn-in exclusion: 2016, the engine's cold-start season, is a
     violent within-train outlier in every market's actual/proj
     ratio distribution (KS D = 0.31/0.20/0.14 vs D <= 0.10 for all
     other train seasons) -- no prior states, different animal. It
     stays in the walk-forward (states need it) but out of shape
     fitting. This, not era drift, was most of pass_yds' v1 failure;
     the remainder still fails the gate, so pass_yds STAYS WITHHELD.

V3 (2026-09-20), red-zone usage -- "assess tendencies and script,
treat realized TDs as the noisy echo":
  TD lambda becomes a train-tuned blend, 0.7 x usage + 0.3 x v2:
  usage-lambda = sum over zones (player share of team zone volume x
  current team zone volume x league conversion in zone) x opponent
  RZ-conversion multiplier + long-TD term (touches outside the 20 x
  credibility-shrunk breakaway rate). Zone conversion gradient is the
  whole point: inside-5 carries score 42%, targets inside-10 38%,
  touches outside the 20 1.3% -- WHERE the touches happen is signal,
  the box-score TD count is the echo. b relaxed 0.40 -> 0.55: the
  causal lambda needs less streak compression. HELD-OUT 2024-25:
  beats v2 on its own pool (log-loss 0.5406 vs 0.5499) and on the
  common pool (0.5446 vs 0.5503; Brier 0.18198 vs 0.18348), buckets
  within ~1.2pp, conservative side. Red-zone data comes from nflverse
  play-by-play at load time; when it is unavailable the engine falls
  back to pure v2 silently -- graceful degradation, never a crash.
"""

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.parquet"
COLS = ["player_id", "player_display_name", "position", "team", "opponent_team", "season", "week", "season_type",
        "attempts", "carries", "targets", "passing_yards", "rushing_yards", "receiving_yards",
        "passing_tds", "rushing_tds", "receiving_tds"]

ALPHA = 1 - 0.5 ** (1 / 4)      # EWMA halflife: 4 games
SEASON_CARRY = 0.6              # state weight surviving a season boundary
CRED_OPP = {"pass_yds": 90, "rush_yds": 45, "rec_yds": 18}   # opportunities before player >> position mean
MIN_PROJ_OPP = {"pass_yds": 20.0, "rush_yds": 10.0, "rec_yds": 4.0}  # proppable-volume threshold
OPP_SHRINK = 0.5
# v2 TD constants. ENV_DAMP / OPP_SHRINK_TD / TD_POWER were chosen on
# TRAIN-ONLY log-loss + bucket calibration (grid in the ledger); the
# held-out gate then verified the single chosen config once.
OPP_SHRINK_TD = 0.5             # opponent TD-defense multiplier shrink
ENV_DAMP = 0.5                  # sqrt damping on the team-environment ratio
ENV_CLAMP = (0.6, 1.5)          # bounds on the (damped) environment multiplier
TIER_CUTS = (8.0, 15.0)         # opportunities/game -> tier 0 / 1 / 2
TIER_K = {0: 5.0, 1: 3.0, 2: 1.5}   # TD credibility k by volume tier
TD_POWER = 0.55                 # b in P(score) = 1 - exp(-a_tier * lam^b) (v3 refit)
TD_MIN_LAMBDA = 0.15            # calibrated-pool floor: no TD opinion below it
BURN_IN_SEASONS = 1             # cold-start seasons excluded from shape fitting
# v3 red-zone usage constants. Zones: c5/c10 = carries by yardline,
# t10/t20 = targets by yardline, o20 = touches outside the 20. League
# conversion defaults are the 2016-2025 pooled empirical rates.
RZ_ZONES = ("c5", "c10", "t10", "t20")
CONV_DEFAULT = {"c5": 0.42, "c10": 0.13, "t10": 0.38, "t20": 0.14, "o20": 0.013}
RZ_OPP_SHRINK = 0.5             # opponent RZ-conversion multiplier shrink
LONG_CRED = 150.0               # o20 touches before a player's own long-TD rate dominates
USAGE_W = 0.7                   # blend weight: w*usage-lambda + (1-w)*v2-lambda (train-tuned)
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
PBP_COLS = ["season", "week", "season_type", "posteam", "defteam", "yardline_100", "play_type",
            "rush_attempt", "pass_attempt", "rush_touchdown", "pass_touchdown", "two_point_attempt",
            "rusher_player_id", "receiver_player_id"]
MARKETS = ("pass_yds", "rush_yds", "rec_yds")
MK_OPP = {"pass_yds": "attempts", "rush_yds": "carries", "rec_yds": "targets"}
MK_YDS = {"pass_yds": "passing_yards", "rush_yds": "rushing_yards", "rec_yds": "receiving_yards"}


class Ewma:
    __slots__ = ("v", "w")
    def __init__(self):
        self.v, self.w = 0.0, 0.0
    def add(self, x):
        self.v = (1 - ALPHA) * self.v + ALPHA * x
        self.w = (1 - ALPHA) * self.w + ALPHA
    def decay_season(self):
        self.v *= 1.0
        self.w *= SEASON_CARRY
    def mean(self, default=0.0):
        return self.v / self.w if self.w > 1e-6 else default


class Engine:
    def __init__(self):
        self.p_opp = defaultdict(Ewma)      # (player, mkt) -> opportunities/game
        self.p_eff = defaultdict(Ewma)      # (player, mkt) -> yards/opportunity
        self.p_td = defaultdict(Ewma)       # player -> (rush+rec) TDs/game
        self.t_vol = defaultdict(Ewma)      # (team, mkt) -> team opportunities/game
        self.d_allow = defaultdict(Ewma)    # (defense, mkt) -> allowed yards/opp
        self.lg_eff = defaultdict(Ewma)     # mkt -> league yards/opp
        self.lg_vol = defaultdict(Ewma)     # mkt -> league team opp/game
        self.lg_td = Ewma()                 # league (rush+rec) TD/starter-game (context only)
        self.pos = {}                       # player -> position
        self.p_team = {}                    # player -> last team seen
        self.pos_eff = defaultdict(Ewma)    # (position, mkt) -> yards/opp
        # ---- v2 TD decomposition states ----
        self.t_td = defaultdict(Ewma)       # team -> offensive skill TDs/game
        self.d_td_allow = defaultdict(Ewma) # defense -> skill TDs allowed/game
        self.p_env_td = defaultdict(Ewma)   # player -> team TDs/game in games HE played
        self.lg_team_td = Ewma()            # league team TDs/game
        self.tier_td = defaultdict(Ewma)    # (position, tier) -> TDs/game tier mean
        # ---- v3 red-zone usage states ----
        self.p_z = defaultdict(Ewma)        # (player, zone) -> zone opportunities/game
        self.p_long = defaultdict(Ewma)     # player -> long (o20) TDs/game
        self.t_z = defaultdict(Ewma)        # (team, zone) -> team zone opportunities/game
        self.lg_z = defaultdict(Ewma)       # zone -> league team zone opp/game
        self.conv = defaultdict(Ewma)       # zone -> league TD conversion in zone
        self.d_rz = defaultdict(Ewma)       # defense -> RZ TDs allowed per RZ opp faced
        self.lg_rz_conv = Ewma()            # league pooled RZ conversion

    def season_boundary(self):
        for d in (self.p_opp, self.p_eff, self.t_vol, self.d_allow, self.lg_eff, self.lg_vol,
                  self.pos_eff, self.t_td, self.d_td_allow, self.p_env_td, self.tier_td,
                  self.p_z, self.p_long, self.t_z, self.lg_z, self.conv, self.d_rz):
            for e in d.values():
                e.decay_season()
        for e in self.p_td.values():
            e.decay_season()
        self.lg_td.decay_season()
        self.lg_team_td.decay_season()
        self.lg_rz_conv.decay_season()

    def volume_tier(self, player):
        """0 = fringe, 1 = rotational, 2 = workhorse; carries+targets/game."""
        opp = self.p_opp[(player, "rush_yds")].mean() + self.p_opp[(player, "rec_yds")].mean()
        return 0 if opp < TIER_CUTS[0] else (1 if opp < TIER_CUTS[1] else 2)

    # ---- prediction (STRICTLY pre-update) ----
    def project(self, player, team, opponent, mkt):
        pos = self.pos.get(player)
        if pos is None:
            return None
        opp_pg = self.p_opp[(player, mkt)]
        if opp_pg.w < 0.35:                          # <~2 games of signal: refuse rather than guess
            return None
        team_vol = self.t_vol[(team, mkt)].mean(self.lg_vol[mkt].mean(30.0))
        lg_vol = self.lg_vol[mkt].mean(30.0)
        share = opp_pg.mean() / max(self.t_vol[(team, mkt)].mean(lg_vol), 1e-6)
        share = min(share, 1.0)
        proj_opp = share * team_vol
        n_opp_seen = opp_pg.w / ALPHA * opp_pg.mean()  # rough opportunity count in state
        k = CRED_OPP[mkt]
        pos_eff = self.pos_eff[(pos, mkt)].mean(self.lg_eff[mkt].mean(7.0))
        eff = (self.p_eff[(player, mkt)].mean(pos_eff) * n_opp_seen + pos_eff * k) / (n_opp_seen + k)
        lg = self.lg_eff[mkt].mean(7.0)
        opp_mult = 1.0 + OPP_SHRINK * ((self.d_allow[(opponent, mkt)].mean(lg) / max(lg, 1e-6)) - 1.0)
        return {"opp": proj_opp, "eff": eff, "yards": proj_opp * eff * opp_mult, "pos": pos}

    def td_components(self, player, team=None, opponent=None):
        """Base credibility-shrunk rate + raw (undamped) env and
        opponent factors, so multiplier strength is tunable on train."""
        e = self.p_td[player]
        if e.w < 0.35:
            return None
        n_games = e.w / ALPHA
        # v2 star-tier credibility: shrink toward the (position, tier)
        # mean with a tier-dependent k -- a workhorse's own rate is
        # trusted almost as-is, a fringe player's barely at all.
        tier = self.volume_tier(player)
        pos = self.pos.get(player)
        target = self.tier_td[(pos, tier)].mean(0.25)
        k = TIER_K[tier]
        base = (e.mean() * n_games + target * k) / (n_games + k)
        lg = self.lg_team_td.mean(2.5)
        env_ratio = 1.0
        if team is not None:
            # v2 causal environment: his base rate was earned inside the
            # TD environment of the games he actually played; rescale it
            # to the CURRENT team's environment. QB injuries, coordinator
            # collapses, and team changes all enter here, immediately.
            team_env = self.t_td[team].mean(lg)
            hist_env = self.p_env_td[player].mean(lg)
            env_ratio = team_env / max(hist_env, 0.25)
        opp_dev = 0.0
        if opponent is not None:
            # v2 opponent layer: TD defenses differ from yardage defenses
            # (red-zone identity), so this uses TDs allowed, not yards.
            opp_dev = (self.d_td_allow[opponent].mean(lg) / max(lg, 1e-6)) - 1.0
        return {"base": base, "env_ratio": env_ratio, "opp_dev": opp_dev, "tier": tier}

    def td_usage_lambda(self, player, team, opponent=None):
        """v3: expected TDs from WHERE the touches happen -- zone usage
        share x current team zone volume x league conversion, opponent-
        adjusted by RZ conversion allowed, plus a credibility-shrunk
        long-TD term. Returns None when red-zone states are thin (no
        pbp data loaded, or a cold player) -- callers fall back to v2."""
        if self.p_z[(player, "o20")].w < 0.35:
            return None
        d_mult = 1.0
        lgc = self.lg_rz_conv.mean(0.25)
        if opponent is not None and self.d_rz[opponent].w > 1e-6:
            d_mult = 1.0 + RZ_OPP_SHRINK * (self.d_rz[opponent].mean(lgc) / max(lgc, 1e-6) - 1.0)
        total = 0.0
        for z in RZ_ZONES:
            tz = self.t_z[(team, z)].mean(self.lg_z[z].mean(1.0))
            share = min(self.p_z[(player, z)].mean() / max(tz, 1e-6), 1.0)
            total += share * tz * self.conv[z].mean(CONV_DEFAULT[z])
        total *= d_mult
        o20 = self.p_z[(player, "o20")].mean()
        n_touch = self.p_z[(player, "o20")].w / ALPHA * o20
        lgl = self.conv["o20"].mean(CONV_DEFAULT["o20"])
        rate = ((self.p_long[player].mean() / max(o20, 1e-6)) * n_touch + lgl * LONG_CRED) / (n_touch + LONG_CRED)
        total += o20 * rate
        return max(total, 0.01)

    def project_td_lambda(self, player, team=None, opponent=None):
        c = self.td_components(player, team, opponent)
        if c is None:
            return None
        # ENV_DAMP/OPP_SHRINK_TD chosen on TRAIN calibration only (see
        # ledger): raw multipliers made high-lambda claims run hot.
        env = min(max(c["env_ratio"] ** ENV_DAMP, ENV_CLAMP[0]), ENV_CLAMP[1])
        lam = c["base"] * env * (1.0 + OPP_SHRINK_TD * c["opp_dev"])
        # v3 blend: where the touches happen (usage) tempered by what
        # they have produced (v2). USAGE_W train-tuned; usage silent ->
        # pure v2, so the engine degrades gracefully without pbp data.
        if team is not None:
            u = self.td_usage_lambda(player, team, opponent)
            if u is not None:
                lam = USAGE_W * u + (1.0 - USAGE_W) * lam
        return max(lam, 0.01)

    # ---- update with a played week ----
    def update_week(self, rows):
        team_opp = defaultdict(float)
        team_yds = defaultdict(float)
        team_td = defaultdict(float)
        opp_of = {}
        # v2: tiers snapshot BEFORE this week's stats enter any state,
        # so tier assignment is strictly walk-forward.
        tiers = {r["player_display_name"]: self.volume_tier(r["player_display_name"]) for r in rows}
        for r in rows:
            team_td[r["team"]] += r["rushing_tds"] + r["receiving_tds"]
            opp_of[r["team"]] = r["opponent_team"]
            for mkt in MARKETS:
                team_opp[(r["team"], mkt)] += r[MK_OPP[mkt]]
                team_yds[(r["opponent_team"], mkt)] += r[MK_YDS[mkt]]   # what the DEFENSE allowed
        for team, td in team_td.items():
            self.t_td[team].add(td)
            self.lg_team_td.add(td)
            self.d_td_allow[opp_of[team]].add(td)
        # ---- v3 red-zone weekly aggregates (zeros count: a team with no
        # inside-5 carries this week must pull its volume state down) ----
        ALL_Z = RZ_ZONES + ("o20",)
        wk_opp = defaultdict(float)
        wk_td = defaultdict(float)
        has_rz = any(r.get(z) for r in rows for z in ALL_Z)
        if has_rz:
            for r in rows:
                for z in ALL_Z:
                    wk_opp[(r["team"], z)] += r.get(z, 0) or 0
                    wk_td[(r["team"], z)] += r.get(z + "_td", 0) or 0
            lg_o = defaultdict(float)
            lg_t = defaultdict(float)
            for (team, z), n in wk_opp.items():
                lg_o[z] += n
                lg_t[z] += wk_td[(team, z)]
            for team in {r["team"] for r in rows}:
                rz_faced = rz_allowed = 0.0
                for z in ALL_Z:
                    self.t_z[(team, z)].add(wk_opp.get((team, z), 0.0))
                    self.lg_z[z].add(wk_opp.get((team, z), 0.0))
                    if z != "o20":
                        rz_faced += wk_opp.get((team, z), 0.0)
                        rz_allowed += wk_td.get((team, z), 0.0)
                if rz_faced >= 3:
                    self.d_rz[opp_of[team]].add(rz_allowed / rz_faced)
            for z in ALL_Z:
                if lg_o[z] >= 10:
                    self.conv[z].add(lg_t[z] / lg_o[z])
            rz_o = sum(lg_o[z] for z in RZ_ZONES)
            if rz_o >= 10:
                self.lg_rz_conv.add(sum(lg_t[z] for z in RZ_ZONES) / rz_o)
        for (team, mkt), n in team_opp.items():
            self.t_vol[(team, mkt)].add(n)
            self.lg_vol[mkt].add(n)
        # defense allowed efficiency this week
        opp_faced = defaultdict(float)
        for r in rows:
            for mkt in MARKETS:
                opp_faced[(r["opponent_team"], mkt)] += r[MK_OPP[mkt]]
        for (dteam, mkt), yds in team_yds.items():
            n = opp_faced[(dteam, mkt)]
            if n >= 5:
                self.d_allow[(dteam, mkt)].add(yds / n)
        played = set()
        teams_played = set()
        for r in rows:
            played.add(r["player_display_name"])
            teams_played.add(r["team"])
        # DNP decay (v1.2 -- v1.1's root defect): a player whose team
        # played without him gets ZEROS into volume and TD states.
        # Without this, injured or role-lost players carry stale
        # featured-role projections forever, polluting the pool and
        # forcing center factors near 0.5.
        for p2, team2 in list(self.p_team.items()):
            if team2 in teams_played and p2 not in played:
                for mkt2 in MARKETS:
                    self.p_opp[(p2, mkt2)].add(0.0)
                self.p_td[p2].add(0.0)
                for z2 in RZ_ZONES + ("o20",):
                    self.p_z[(p2, z2)].add(0.0)
                self.p_long[p2].add(0.0)
        for r in rows:
            p = r["player_display_name"]
            self.pos[p] = r["position"]
            self.p_team[p] = r["team"]
            for mkt in MARKETS:
                n = r[MK_OPP[mkt]]
                self.p_opp[(p, mkt)].add(n)
                if n >= 3:
                    eff = r[MK_YDS[mkt]] / n
                    self.p_eff[(p, mkt)].add(eff)
                    self.pos_eff[(r["position"], mkt)].add(eff)
                    self.lg_eff[mkt].add(eff)
            self.p_td[p].add(r["rushing_tds"] + r["receiving_tds"])
            # v2: environment his production was earned in, and the
            # (position, tier) mean the credibility shrink targets.
            self.p_env_td[p].add(team_td[r["team"]])
            self.tier_td[(r["position"], tiers[p])].add(r["rushing_tds"] + r["receiving_tds"])
            if has_rz:
                for z in RZ_ZONES + ("o20",):
                    self.p_z[(p, z)].add(r.get(z, 0) or 0)
                self.p_long[p].add(r.get("o20_td", 0) or 0)


def load_seasons(first, last):
    frames = []
    for s in range(first, last + 1):
        df = pd.read_parquet(STATS_URL.format(season=s), columns=COLS)
        df = df[df["season_type"] == "REG"].fillna(0)
        frames.append(df)
        print(f"  loaded {s}: {len(df)} player-weeks")
    return pd.concat(frames, ignore_index=True)


def load_rz(first, last, cache_dir=None):
    """Aggregate nflverse play-by-play into per-player-week red-zone
    usage: zone opportunities and TDs, keyed by gsis player_id. REG
    season, 2-pt plays excluded. cache_dir keeps the ~20MB/season pbp
    downloads across runs (env RZ_CACHE_DIR overrides)."""
    import urllib.request
    cache_dir = cache_dir or os.environ.get("RZ_CACHE_DIR") or "/tmp/nfl_pbp_cache"
    os.makedirs(cache_dir, exist_ok=True)
    frames = []
    for season in range(first, last + 1):
        path = os.path.join(cache_dir, f"pbp_{season}.parquet")
        if not os.path.exists(path):
            urllib.request.urlretrieve(PBP_URL.format(season=season), path)
        pbp = pd.read_parquet(path, columns=PBP_COLS)
        pbp = pbp[(pbp["season_type"] == "REG") & (pbp["two_point_attempt"].fillna(0) == 0)]
        parts = []
        for idc, att, tdc, is_rush in (("rusher_player_id", "rush_attempt", "rush_touchdown", True),
                                       ("receiver_player_id", "pass_attempt", "pass_touchdown", False)):
            d = pbp[(pbp[att].fillna(0) == 1) & pbp[idc].notna()].copy()
            y = d["yardline_100"]
            if is_rush:
                d["z"] = np.where(y <= 5, "c5", np.where(y <= 10, "c10", "o20"))
            else:
                d["z"] = np.where(y <= 10, "t10", np.where(y <= 20, "t20", "o20"))
            d["td"] = d[tdc].fillna(0).astype(int)
            d = d.rename(columns={idc: "player_id"})
            parts.append(d[["season", "week", "player_id", "z", "td"]])
        d = pd.concat(parts, ignore_index=True)
        g = d.groupby(["season", "week", "player_id", "z"]).agg(opp=("td", "size"), td=("td", "sum")).reset_index()
        wide = g.pivot_table(index=["season", "week", "player_id"], columns="z",
                             values=["opp", "td"], fill_value=0)
        wide.columns = [z if k == "opp" else z + "_td" for k, z in wide.columns]
        wide = wide.reset_index()
        for c in [z for z in RZ_ZONES + ("o20",)] + [z + "_td" for z in RZ_ZONES + ("o20",)]:
            if c not in wide.columns:
                wide[c] = 0
        frames.append(wide)
        print(f"  rz {season}: {len(wide)} player-weeks")
    return pd.concat(frames, ignore_index=True)


def merge_rz(data, rz):
    """Left-join stats player-weeks with red-zone usage; missing = 0."""
    zcols = [z for z in RZ_ZONES + ("o20",)] + [z + "_td" for z in RZ_ZONES + ("o20",)]
    out = data.merge(rz[["season", "week", "player_id"] + zcols],
                     on=["season", "week", "player_id"], how="left")
    out[zcols] = out[zcols].fillna(0)
    return out


def walk_forward(first=2016, last=2025, data=None):
    """Predictions strictly before each week's update. Returns rows of
    (season, week, player, market, proj, actual, td_lambda, scored)."""
    data = data if data is not None else load_seasons(first, last)
    eng = Engine()
    preds = []
    from collections import deque
    trail = defaultdict(lambda: deque(maxlen=4))   # (player, mkt) -> last 4 actuals
    for season in range(first, last + 1):
        if season > first:
            eng.season_boundary()
        sdf = data[data["season"] == season]
        for week in sorted(sdf["week"].unique()):
            wdf = sdf[sdf["week"] == week]
            rows = wdf.to_dict("records")
            for r in rows:
                p = r["player_display_name"]
                for mkt in MARKETS:
                    pr = eng.project(p, r["team"], r["opponent_team"], mkt)
                    if pr and pr["opp"] >= MIN_PROJ_OPP[mkt]:
                        t4 = trail[(p, mkt)]
                        preds.append({"season": season, "week": week, "player": p, "mkt": mkt,
                                      "proj": pr["yards"], "opp_proj": pr["opp"], "actual": r[MK_YDS[mkt]],
                                      "baseline": (sum(t4) / len(t4)) if len(t4) >= 2 else None})
                c = eng.td_components(p, r["team"], r["opponent_team"])
                lam = eng.project_td_lambda(p, r["team"], r["opponent_team"])
                if lam is not None and r["position"] in ("RB", "WR", "TE") and max(lam, c["base"]) >= TD_MIN_LAMBDA:
                    u = eng.td_usage_lambda(p, r["team"], r["opponent_team"])
                    preds.append({"season": season, "week": week, "player": p, "mkt": "anytime_td",
                                  "proj": lam, "tier": c["tier"], "td_base": c["base"],
                                  "td_env": c["env_ratio"], "td_opp": c["opp_dev"],
                                  "td_usage": u if u is not None else np.nan,
                                  "actual": float(r["rushing_tds"] + r["receiving_tds"] > 0)})
            for r in rows:
                for mkt in MARKETS:
                    if r[MK_OPP[mkt]] >= 1:
                        trail[(r["player_display_name"], mkt)].append(r[MK_YDS[mkt]])
            eng.update_week(rows)
    return pd.DataFrame(preds), eng


def _fit_a(lam_pow, actual):
    """Bisect a s.t. mean(1 - exp(-a * lam^b)) on this train subset
    equals its realized scoring rate."""
    lo, hi = 0.05, 5.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if (1 - np.exp(-mid * lam_pow)).mean() > actual.mean():
            hi = mid
        else:
            lo = mid
    return float((lo + hi) / 2)


def build_shape(train):
    """Frozen per-market ECDF of actual/RAW-proj ratios + the train
    median ratio used to re-center point estimates (raw projections
    run structurally hot: EWMAs describe healthy featured games while
    reality includes role losses and blowouts -- the ECDF absorbs
    that for probabilities, and the median ratio corrects the quoted
    point estimate). TD gets a train-fit per-tier power recalibration
    P(score) = 1 - exp(-a_tier * lam^b) -- see module docstring."""
    shapes = {"_center": {}, "_td_a": {}, "_td_b": TD_POWER, "_cuts": {}}
    # v2 burn-in exclusion: the engine's first BURN_IN_SEASONS have no
    # prior states and their actual/proj ratio distributions are violent
    # within-train outliers (2016: KS D = 0.31/0.20/0.14 by market vs
    # D <= 0.10 for every later train season). They feed the walk-forward
    # states but not the frozen shapes.
    train = train[train["season"] > train["season"].min() + BURN_IN_SEASONS - 1]
    for mkt in MARKETS:
        t = train[(train["mkt"] == mkt) & (train["proj"] > 0)]
        # Stratify by projected volume tercile: fringe roles and
        # workhorses have different outcome shapes (v1.1 finding).
        cuts = t["opp_proj"].quantile([1.0 / 3, 2.0 / 3]).values
        shapes["_cuts"][mkt] = cuts
        strata = [t[t["opp_proj"] <= cuts[0]],
                  t[(t["opp_proj"] > cuts[0]) & (t["opp_proj"] <= cuts[1])],
                  t[t["opp_proj"] > cuts[1]]]
        for st, sel in enumerate(strata):
            ratios = np.sort((sel["actual"] / sel["proj"]).values)
            shapes[(mkt, st)] = ratios
            shapes["_center"][(mkt, st)] = float(np.median(ratios)) if len(ratios) else 1.0
        allr = np.sort((t["actual"] / t["proj"]).values)
        shapes[mkt] = allr
        shapes["_center"][mkt] = float(np.median(allr))
    td = train[train["mkt"] == "anytime_td"]
    if len(td) > 500:
        lam_pow = td["proj"].values ** TD_POWER
        pooled = _fit_a(lam_pow, td["actual"].values)
        shapes["_td_a"]["pooled"] = pooled
        # v2 star-tier recalibration: a_tier levels each tier's claimed
        # rate; the shared b < 1 compresses hot-streak lambdas that
        # mean-revert (the pooled scalar of v1 could not do both).
        for tier in (0, 1, 2):
            sub = td[td["tier"] == tier] if "tier" in td.columns else td.iloc[0:0]
            shapes["_td_a"][tier] = (_fit_a(sub["proj"].values ** TD_POWER, sub["actual"].values)
                                     if len(sub) > 500 else pooled)
    return shapes


def point_estimate(shapes, mkt, raw_proj):
    return raw_proj * shapes["_center"].get(mkt, 1.0)


def prob_score(shapes, td_lambda, tier=None):
    a = shapes.get("_td_a", {})
    a = a.get(tier, a.get("pooled", 1.0))
    return float(1 - np.exp(-a * td_lambda ** shapes.get("_td_b", 1.0)))


def _stratum(shapes, mkt, opp_proj):
    cuts = shapes.get("_cuts", {}).get(mkt)
    if cuts is None or opp_proj is None:
        return None
    return 0 if opp_proj <= cuts[0] else (1 if opp_proj <= cuts[1] else 2)


def prob_over(shapes, mkt, proj, line, opp_proj=None):
    st = _stratum(shapes, mkt, opp_proj)
    ratios = shapes.get((mkt, st)) if st is not None else None
    if ratios is None or len(ratios) < 200:
        ratios = shapes.get(mkt)
    if ratios is None or len(ratios) < 200 or proj <= 0:
        return None
    return float(1.0 - np.searchsorted(ratios, line / proj, side="left") / len(ratios))


def calibration_report(preds, shapes, test_seasons):
    test = preds[preds["season"].isin(test_seasons)]
    print(f"\n===== HELD-OUT CALIBRATION ({test_seasons}) =====")
    for mkt in MARKETS:
        t = test[test["mkt"] == mkt]
        if not len(t):
            continue
        print(f"\n{mkt} (n={len(t)}, median proj {t['proj'].median():.1f}):")
        print("  synthetic line   claimed P(over)   actual hit rate")
        for mult in (0.7, 0.85, 1.0, 1.15, 1.3):
            claimed = [prob_over(shapes, mkt, p, p * mult, o) for p, o in zip(t["proj"], t["opp_proj"])]
            actual = (t["actual"] > t["proj"] * mult).mean()
            print(f"    {mult:4.2f} x proj      {np.mean(claimed):.3f}             {actual:.3f}")
        centered = pd.Series([p * shapes["_center"].get((mkt, _stratum(shapes, mkt, o)), shapes["_center"][mkt])
                              for p, o in zip(t["proj"], t["opp_proj"])], index=t.index)
        mae = (t["actual"] - centered).abs().mean()
        tb = t[t["baseline"].notna()]
        base_mae = (tb["actual"] - tb["baseline"]).abs().mean() if len(tb) else float("nan")
        print(f"  centered point MAE {mae:.1f} yds  vs trailing-4 baseline {base_mae:.1f} yds  "
              f"(center x{shapes['_center'][mkt]:.3f})")
    td = test[test["mkt"] == "anytime_td"]
    if len(td):
        td = td.copy()
        td["p"] = [prob_score(shapes, lam, t) for lam, t in zip(td["proj"], td.get("tier", pd.Series(index=td.index)))]
        print(f"\nanytime_td (n={len(td)}):")
        print("  bucket        claimed   actual    n")
        for lo, hi in ((0.1, 0.25), (0.25, 0.4), (0.4, 0.55), (0.55, 0.75), (0.75, 0.95)):
            b = td[(td["p"] >= lo) & (td["p"] < hi)]
            if len(b) > 30:
                print(f"  {lo:.2f}-{hi:.2f}     {b['p'].mean():.3f}     {b['actual'].mean():.3f}   {len(b)}")
        if "tier" in td.columns:
            print("  by tier       claimed   actual    n")
            for tier in (0, 1, 2):
                b = td[td["tier"] == tier]
                if len(b) > 30:
                    print(f"  tier {tier}        {b['p'].mean():.3f}     {b['actual'].mean():.3f}   {len(b)}")


# ---------------- live pipeline API (watch mode) ----------------

CALIBRATED_MARKETS = {"rush_yds", "rec_yds", "anytime_td"}
# pass_yds STILL WITHHELD after the v2 retry: burn-in exclusion
# (which is what v1 misdiagnosed as era drift) closed the held-out
# over-claim from +5-8pp to +2-5pp -- better, still a fail. The
# remainder is a genuine train/test shape mismatch no train-only fit
# can see. It stays out until a revision passes the same gate -- the
# biggest market sitting out IS the point.

PROP_MARKET_MAP = {"player_rush_yds": "rush_yds", "player_reception_yds": "rec_yds",
                   "player_anytime_td": "anytime_td", "player_pass_yds": "pass_yds"}


def load_shapes(path=None):
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "player_shape_ecdf.npz")
    z = np.load(path, allow_pickle=False)
    shapes = {"_center": {}, "_cuts": {}, "_td_b": float(z["td_b"][0]),
              "_td_a": {t: float(z["td_a"][t]) for t in range(3)}}
    shapes["_td_a"]["pooled"] = float(z["td_a"][3])
    for m in MARKETS:
        shapes[m] = z[m]
        shapes["_cuts"][m] = z[m + "_cuts"]
        for st in range(3):
            shapes[(m, st)] = z[m + "_s%d" % st]
            shapes["_center"][(m, st)] = float(z[m + "_c%d" % st][0])
        shapes["_center"][m] = float(np.median(z[m]))
    return shapes


def _norm_name(name):
    """Lowercase, punctuation-free; suffix tokens dropped separately by
    the index builder. 'Brian Thomas Jr' (The Odds API) must find
    'Brian Thomas Jr.' (nflverse)."""
    return " ".join(str(name).lower().replace(".", "").replace(",", "").split())


_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _suffixless(norm):
    parts = norm.split()
    return " ".join(parts[:-1]) if len(parts) > 2 and parts[-1] in _SUFFIXES else None


class LiveProjector:
    """Current-season projector for the odds watch: walks prior plus
    current season stats into an Engine, answers P(over line) and
    P(scores) for CALIBRATED markets only. Team-model-blind, like
    everything in this module."""

    def __init__(self, season, data=None):
        self.shapes = load_shapes()
        self.eng = Engine()
        if data is None:
            data = load_seasons(season - 1, season)
            try:
                data = merge_rz(data, load_rz(season - 1, season))
                print("[engine] red-zone usage merged")
            except Exception as e:                        # noqa: BLE001
                print(f"[engine] rz unavailable ({e}); TD lambda falls back to v2")
        first = data["season"].min()
        for yr in sorted(data["season"].unique()):
            if yr != first:
                self.eng.season_boundary()
            sdf = data[data["season"] == yr]
            for wk in sorted(sdf["week"].unique()):
                self.eng.update_week(sdf[sdf["week"] == wk].to_dict("records"))
        # Name index: books and nflverse disagree on suffix punctuation.
        # Ambiguous suffixless keys (two distinct players) are dropped.
        self._names = {}
        drop = set()
        for name in self.eng.p_team:
            for key in (_norm_name(name), _suffixless(_norm_name(name))):
                if key is None:
                    continue
                if key in self._names and self._names[key] != name:
                    drop.add(key)
                self._names[key] = name
        for key in drop:
            self._names.pop(key, None)

    def prop_opinion(self, player, home_team, away_team, prop_market, line):
        mkt = PROP_MARKET_MAP.get(prop_market)
        if mkt is None or mkt not in CALIBRATED_MARKETS:
            return None
        if player not in self.eng.p_team:
            key = _norm_name(player)
            player = self._names.get(key) or self._names.get(_suffixless(key) or "") or player
        team = self.eng.p_team.get(player)
        if team not in (home_team, away_team):
            return None            # unknown player or name mismatch: silence beats guessing
        opponent = away_team if team == home_team else home_team
        if mkt == "anytime_td":
            c = self.eng.td_components(player, team, opponent)
            lam = self.eng.project_td_lambda(player, team, opponent)
            if lam is None or max(lam, c["base"]) < TD_MIN_LAMBDA:
                # Below the calibrated pool (the held-out gate was fit and
                # verified on this floor). b < 1 inflates tiny lambdas, so
                # out-of-pool extrapolation flatters exactly the fringe
                # players the edge board already over-surfaces. Silence.
                return None
            tier = self.eng.volume_tier(player)
            return {"market": mkt, "p_score": round(prob_score(self.shapes, lam, tier), 4), "kind": "score"}
        pr = self.eng.project(player, team, opponent, mkt)
        if pr is None or pr["opp"] < MIN_PROJ_OPP[mkt] or line is None:
            return None
        p = prob_over(self.shapes, mkt, pr["yards"], line, pr["opp"])
        if p is None:
            return None
        st = _stratum(self.shapes, mkt, pr["opp"])
        center = self.shapes["_center"].get((mkt, st), self.shapes["_center"].get(mkt, 1.0))
        return {"market": mkt, "p_over": round(p, 4), "median": round(pr["yards"] * center, 1), "kind": "yards"}


def main():
    data = load_seasons(2016, 2025)
    try:
        data = merge_rz(data, load_rz(2016, 2025))
        print("red-zone usage merged")
    except Exception as e:                                # noqa: BLE001
        print(f"[rz] unavailable ({e}); walk runs v2-only")
    preds, _ = walk_forward(2016, 2025, data=data)
    train = preds[preds["season"] <= 2023]
    shapes = build_shape(train)
    print(f"\nwalk-forward predictions: {len(preds)} | train {len(train)} | shapes: " +
          ", ".join(f"{m}:{len(shapes[m])}" for m in MARKETS))
    calibration_report(preds, shapes, [2024, 2025])
    # persist shapes for the live pipeline
    save = {"td_b": np.array([shapes["_td_b"]]),
            "td_a": np.array([shapes["_td_a"].get(t, shapes["_td_a"].get("pooled", 1.0))
                              for t in (0, 1, 2, "pooled")])}
    for m in MARKETS:
        save[m] = shapes[m]
        save[m + "_cuts"] = shapes["_cuts"][m]
        for st in range(3):
            save[m + "_s%d" % st] = shapes[(m, st)]
            save[m + "_c%d" % st] = np.array([shapes["_center"][(m, st)]])
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "player_shape_ecdf.npz"), **save)
    print("\nshapes frozen to model/player_shape_ecdf.npz (train <= 2023)")


if __name__ == "__main__":
    main()
