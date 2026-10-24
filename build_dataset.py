"""
Beating The Bookie - Layer 1: Data
Builds a clean player-game dataset with pre-game-only features for 3PM props.

Rule: every feature for a game uses ONLY games played before that date (no look-ahead).
Source: sportsdataverse (ESPN box scores hosted on GitHub).
Season label = year the season ends (2026 = 2025-26).
"""
import pandas as pd
import numpy as np

SEASONS = [2024, 2025, 2026]
URL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
       "espn_nba_player_boxscores/player_box_{s}.parquet")
SEASON_TYPES = {2: "regular", 3: "playoffs", 5: "play-in"}

# Empirical-Bayes prior for 3P%: shrink each player's 3P% toward league average
# by adding PRIOR_ATT phantom attempts at the league rate.
PRIOR_ATT = 150


def load(seasons=SEASONS):
    frames = [pd.read_parquet(URL.format(s=s)) for s in seasons]
    d = pd.concat(frames, ignore_index=True)
    d = d.rename(columns={
        "athlete_id": "player_id", "athlete_display_name": "player",
        "team_abbreviation": "team", "opponent_team_abbreviation": "opp",
        "three_point_field_goals_made": "fg3m",
        "three_point_field_goals_attempted": "fg3a",
    })
    d["game_date"] = pd.to_datetime(d["game_date"])
    d["season_type"] = d["season_type"].map(SEASON_TYPES)
    d["played"] = (~d["did_not_play"].astype(bool)) & (d["minutes"] > 0)
    keep = ["game_id", "season", "season_type", "game_date", "player_id", "player",
            "team", "opp", "home_away", "starter", "played", "reason",
            "minutes", "fg3m", "fg3a", "points", "rebounds", "assists"]
    return d[keep].sort_values(["player_id", "game_date"]).reset_index(drop=True)


def player_features(d):
    p = d[d["played"]].copy()
    g = p.groupby(["player_id", "season"], group_keys=False)

    # Rolling means of PRIOR games only (shift(1) excludes the current game)
    for col in ["minutes", "fg3a", "fg3m"]:
        for n in [5, 10]:
            p[f"{col}_l{n}"] = g[col].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
        p[f"{col}_szn"] = g[col].transform(lambda s: s.shift(1).expanding().mean())

    p["games_prior"] = g.cumcount()
    p["fg3a_per_min_szn"] = (g["fg3a"].transform(lambda s: s.shift(1).cumsum())
                             / g["minutes"].transform(lambda s: s.shift(1).cumsum()))

    # Regressed 3P%: (prior makes + league_rate*PRIOR_ATT) / (prior att + PRIOR_ATT)
    league = (p.groupby("season")["fg3m"].sum() / p.groupby("season")["fg3a"].sum())
    p["league_3p"] = p["season"].map(league)
    cm = g["fg3m"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    ca = g["fg3a"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    p["fg3_pct_reg"] = (cm + p["league_3p"] * PRIOR_ATT) / (ca + PRIOR_ATT)

    # Rest days since this player's previous game
    p["rest_days"] = p.groupby("player_id")["game_date"].diff().dt.days
    return p


def opponent_features(d):
    # Team 3PA allowed per game, season to date, prior games only
    t = (d[d["played"]].groupby(["season", "game_id", "game_date", "opp"])["fg3a"]
         .sum().reset_index().rename(columns={"fg3a": "fg3a_allowed"}))
    t = t.sort_values(["opp", "game_date"])
    t["opp_fg3a_allowed_szn"] = (t.groupby(["opp", "season"])["fg3a_allowed"]
                                 .transform(lambda s: s.shift(1).expanding().mean()))
    lg = t.groupby("season")["fg3a_allowed"].mean()
    t["opp_fg3a_factor"] = t["opp_fg3a_allowed_szn"] / t["season"].map(lg)
    return t[["game_id", "opp", "opp_fg3a_allowed_szn", "opp_fg3a_factor"]]


def build():
    d = load()
    p = player_features(d)
    p = p.merge(opponent_features(d), on=["game_id", "opp"], how="left")
    # Naive baseline projection (Layer 2 will replace this)
    p["proj_fg3m_naive"] = (p["minutes_l10"] * p["fg3a_per_min_szn"]
                            * p["opp_fg3a_factor"].fillna(1) * p["fg3_pct_reg"])
    return d, p


if __name__ == "__main__":
    raw, feats = build()
    feats.to_parquet("player_games_features.parquet", index=False)
    print(f"Raw rows: {len(raw):,} | Played rows with features: {len(feats):,}")
    print(feats.groupby(["season", "season_type"]).size())
