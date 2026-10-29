"""
Beating The Bookie - Layer 2: projection model for 3PM (v1).

Rules
- Every feature uses only games before the game being projected (shift(1) everywhere).
- Unlike Layer 1, history carries ACROSS seasons with exponential decay, so early-season
  projections aren't built from 2-3 games.
- Data split (fixed, never changed after the fact):
    train      : 2023-24 and 2024-25 (all season types)
    validation : 2025-26 regular season before 1 Feb 2026  (used for all model choices)
    holdout    : 2025-26 from 1 Feb 2026 onward            (touched once, at the end of Layer 2)
- Comparison set = Layer 1 test definition: regular season, 10+ prior games this season,
  20+ minutes average over the last 10 (pre-game). Early-season set = games 1-10.

Metrics: MAE (Layer 1 target 1.03) plus RMSE, Poisson deviance and ladder log loss
(P(1+) .. P(4+)), because bets are priced off the distribution, not the median.
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy import stats

import build_dataset as b1

SEASONS = [2023, 2024, 2025, 2026]
VAL_END = pd.Timestamp("2026-02-01")
LADDER = [1, 2, 3, 4]


# ---------- decayed history helpers (all exclude the current game) ----------
def decayed_sum(x, hl):
    a = 0.5 ** (1 / hl)
    y = lfilter([1.0], [1.0, -a], np.nan_to_num(x.astype(float)))
    return pd.Series(np.r_[0.0, y[:-1]], index=x.index)       # shift: prior games only


def by_player(p, col, hl):
    return p.groupby("player_id", group_keys=False)[col].apply(lambda s: decayed_sum(s, hl))


def features():
    d = b1.load(SEASONS)
    lines = pd.read_parquet("game_lines.parquet")[["game_id", "total", "home_fav_by",
                                                   "home_implied_pts", "away_implied_pts"]]
    p = d[d["played"]].copy().sort_values(["player_id", "game_date"])
    p["one"] = 1.0
    p["starter_f"] = p["starter"].astype(float)

    for hl in (5, 20):
        w = by_player(p, "one", hl)
        p[f"min_ewm{hl}"] = by_player(p, "minutes", hl) / w
        p[f"fg3a_ewm{hl}"] = by_player(p, "fg3a", hl) / w
        p[f"start_ewm{hl}"] = by_player(p, "starter_f", hl) / w
        p[f"rate_ewm{hl}"] = by_player(p, "fg3a", hl) / by_player(p, "minutes", hl)

    # 3P% with decayed makes/attempts (half-life 82 games) shrunk to league average
    lg = (p.groupby("season")["fg3m"].sum() / p.groupby("season")["fg3a"].sum())
    p["league_3p"] = p["season"].map(lg)
    mk, at = by_player(p, "fg3m", 82), by_player(p, "fg3a", 82)
    p["pct_reg"] = (mk + p["league_3p"] * b1.PRIOR_ATT) / (at + b1.PRIOR_ATT)
    p["att_hist"] = at

    g = p.groupby(["player_id", "season"])
    p["games_prior"] = g.cumcount()
    p["career_games"] = p.groupby("player_id").cumcount()
    p["last_min"] = p.groupby("player_id")["minutes"].shift(1)
    p["last_start"] = p.groupby("player_id")["starter_f"].shift(1)
    p["rest_days"] = p.groupby("player_id")["game_date"].diff().dt.days.clip(upper=10)
    p["minutes_l10"] = g["minutes"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    p["fg3m_szn"] = g["fg3m"].transform(lambda s: s.shift(1).expanding().mean())
    p["home"] = (p["home_away"] == "home").astype(float)

    p = p.merge(b1.opponent_features(d), on=["game_id", "opp"], how="left")
    p = p.merge(lines, on="game_id", how="left")
    p["team_implied"] = np.where(p.home == 1, p.home_implied_pts, p.away_implied_pts)
    p["team_fav_by"] = np.where(p.home == 1, p.home_fav_by, -p.home_fav_by)
    p["abs_spread"] = p.home_fav_by.abs()
    return p


FEATS = ["min_ewm5", "min_ewm20", "fg3a_ewm5", "fg3a_ewm20", "rate_ewm5", "rate_ewm20",
         "start_ewm5", "start_ewm20", "pct_reg", "att_hist", "last_min", "last_start",
         "rest_days", "home", "games_prior", "career_games", "opp_fg3a_factor",
         "total", "team_implied", "team_fav_by", "abs_spread"]


def structured(p):
    """Interpretable model: minutes x attempt rate x opponent x regressed 3P%."""
    mins = 0.6 * p.min_ewm5 + 0.4 * p.min_ewm20
    rate = 0.35 * p.rate_ewm5 + 0.65 * p.rate_ewm20
    return mins * rate * p.opp_fg3a_factor.fillna(1).clip(0.85, 1.15) * p.pct_reg


def nb_ladder(mu, r):
    """P(X >= k) for k in LADDER under negative binomial with mean mu, size r."""
    mu = np.clip(mu, 1e-3, None)
    pr = r / (r + mu)
    return np.column_stack([stats.nbinom.sf(k - 1, r, pr) for k in LADDER])


def fit_r(mu, y):
    """Method-of-moments NB size from residuals: var = mu + mu^2 / r."""
    excess = ((y - mu) ** 2 - mu).sum() / (mu ** 2).sum()
    return 1 / max(excess, 1e-3)


def evaluate(name, mu, y, r):
    mu, y = np.clip(np.asarray(mu, float), 1e-3, None), np.asarray(y, float)
    P = np.clip(nb_ladder(mu, r), 1e-6, 1 - 1e-6)
    hit = np.column_stack([(y >= k) for k in LADDER])
    ll = -(hit * np.log(P) + (1 - hit) * np.log(1 - P)).mean(axis=0)
    dev = 2 * np.mean(np.where(y > 0, y * np.log(y / mu), 0) - (y - mu))
    return dict(model=name, n=len(y), MAE=np.abs(mu - y).mean(), RMSE=np.sqrt(((mu - y) ** 2).mean()),
                pois_dev=dev, mean_pred=mu.mean(), mean_act=y.mean(),
                **{f"LL_{k}+": v for k, v in zip(LADDER, ll)})


if __name__ == "__main__":
    import lightgbm as lgb

    p = features()
    reg = p.season_type == "regular"
    train = p[p.season.isin([2024, 2025]) & (p.career_games >= 5)]
    val = p[(p.season == 2026) & reg & (p.game_date < VAL_END)]
    cmp_mask = (val.games_prior >= 10) & (val.minutes_l10 >= 20)
    early_mask = (val.games_prior < 10) & (val.min_ewm20 >= 20) & (val.career_games >= 20)

    gbm = lgb.LGBMRegressor(objective="poisson", n_estimators=600, learning_rate=0.03,
                            num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.8, verbose=-1)
    gbm.fit(train[FEATS], train.fg3m)

    l1 = pd.read_parquet("player_games_features.parquet", columns=["game_id", "player_id", "proj_fg3m_naive"])
    val = val.merge(l1, on=["game_id", "player_id"], how="left").set_index(val.index)
    preds = {
        "season avg (L1)": val.fg3m_szn,
        "naive (L1)": val.proj_fg3m_naive,
        "structured v1": structured(val),
        "GBM v1": pd.Series(gbm.predict(val[FEATS]), index=val.index),
    }
    r_struct = fit_r(structured(train).clip(0.01).values, train.fg3m.values)
    r_gbm = fit_r(np.clip(gbm.predict(train[FEATS]), 0.01, None), train.fg3m.values)
    print(f"NB size r: structured {r_struct:.1f}, GBM {r_gbm:.1f}")

    for label, m in [("COMPARISON SET (10+ prior games, 20+ min)", cmp_mask),
                     ("EARLY SEASON (games 1-10, established players)", early_mask)]:
        rows = []
        for k, mu in preds.items():
            ok = m & mu.notna() & np.isfinite(mu)
            r = r_gbm if "GBM" in k else r_struct
            rows.append(evaluate(k, mu[ok], val.fg3m[ok], r))
        print(f"\n{label}\n", pd.DataFrame(rows).round(3).to_string(index=False))

    imp = pd.Series(gbm.booster_.feature_importance("gain"), FEATS).sort_values(ascending=False)
    print("\nGBM feature gain share:\n", (imp / imp.sum()).round(3).head(12).to_string())
    p.to_parquet("player_games_l2.parquet", index=False)
