"""
Beating The Bookie - Layer 2b: absence model (who gains when a teammate sits).

Absent teammate = played for the same team in one of its last 10 games this season, doesn't play
today, and his next game (if any) is still for this team (so trades and departures don't count).
For each player-game we add what walked out the door:
  abs_min / abs_fg3a / abs_pts / abs_start  : summed recent minutes, 3PA, points, starter rate of
                                              absent teammates (decayed averages, half-life 10 games)
  abs_min_pos                               : absent minutes at the player's own position (G/F/C)
  back_min                                  : minutes of teammates who return today after missing
                                              the previous game (their return takes shots back)
Training uses who actually sat. Live use needs absences known before tip, so the bet365 check
below only counts absences already listed Out/Doubtful on the 10:00 UTC injury report.

Split as in projection_model.py: train 2023-24 + 2024-25, validate 2025-26 before 1 Feb.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

import build_dataset as b1
import projection_model as pm

HL = 10
WINDOW = 10
ROT = 15          # only teammates averaging 15+ minutes count as absent / returning
K_SHRINK = 3      # pair responses are shrunk toward 0 by K_SHRINK pseudo-games
POS = {"G": "G", "PG": "G", "SG": "G", "F": "F", "SF": "F", "PF": "F", "C": "C"}
ABS = ["abs_min", "abs_fg3a", "abs_pts", "abs_start", "abs_min_pos", "back_min"]
PAIR = ["pair_min", "pair_3pa", "pair_n"]
FEATS2 = pm.FEATS + ABS
FEATS3 = pm.FEATS + ABS + PAIR
URL = b1.URL


def positions(seasons):
    pos = pd.concat([pd.read_parquet(URL.format(s=s), columns=["athlete_id", "game_date",
                                                               "athlete_position_abbreviation"]) for s in seasons])
    pos = pos.sort_values("game_date").groupby("athlete_id").athlete_position_abbreviation.last()
    return pos.map(POS).fillna("F").rename("pos")


def absent_table(d):
    """One row per (game, team, absent teammate) with his pre-game decayed averages."""
    p = d[d.played].sort_values(["player_id", "game_date"]).copy()
    p["starter_f"] = p.starter.astype(float)
    for c, src in [("m", "minutes"), ("a3", "fg3a"), ("pt", "points"), ("st", "starter_f")]:
        p[c] = p.groupby("player_id")[src].transform(lambda s: s.ewm(halflife=HL).mean())   # incl. that game
    p["next_team"] = p.groupby("player_id").team.shift(-1)
    tg = d[["season", "team", "game_id", "game_date"]].drop_duplicates(["team", "game_id"]).sort_values("game_date")
    tg["tg"] = tg.groupby(["season", "team"]).cumcount()
    p = p.merge(tg[["team", "game_id", "tg"]], on=["team", "game_id"])
    out, back = [], []
    for (season, team), games in tg.groupby(["season", "team"]):
        pt = p[(p.season == season) & (p.team == team)]
        by_tg = {g: set(x.player_id) for g, x in pt.groupby("tg")}
        last = {}                                        # player -> his latest played row so far
        for g in games.tg:
            today = by_tg.get(g, set())
            gid = games.loc[games.tg == g, "game_id"].iloc[0]
            for pid, r in last.items():
                if pid in today or g - r.tg > WINDOW:
                    continue
                if pd.notna(r.next_team) and r.next_team != team:
                    continue                             # traded / left
                if r.m >= ROT:
                    out.append((gid, team, pid, r.m, r.a3, r.pt, r.st))
            prev = by_tg.get(g - 1, set())
            for pid in today - prev:
                if pid in last and g - 1 - last[pid].tg <= WINDOW and g > 0 and last[pid].m >= ROT:
                    back.append((gid, team, pid, last[pid].m))
            for r in pt[pt.tg == g].itertuples():
                last[r.player_id] = r
    a = pd.DataFrame(out, columns=["game_id", "team", "absent_id", "m", "a3", "pt", "st"])
    b = pd.DataFrame(back, columns=["game_id", "team", "back_id", "m"])
    return a, b


def add_absence(p, a, b, pos):
    a = a.merge(pos, left_on="absent_id", right_index=True, how="left")
    agg = a.groupby(["game_id", "team"]).agg(abs_min=("m", "sum"), abs_fg3a=("a3", "sum"),
                                              abs_pts=("pt", "sum"), abs_start=("st", "sum")).reset_index()
    bypos = a.groupby(["game_id", "team", "pos"]).m.sum().rename("abs_min_pos").reset_index()
    bk = b.groupby(["game_id", "team"]).m.sum().rename("back_min").reset_index()
    p = p.drop(columns=[c for c in ABS + ["pos"] if c in p], errors="ignore")
    p = p.merge(pos, left_on="player_id", right_index=True, how="left")
    p = p.merge(agg, on=["game_id", "team"], how="left").merge(bypos, on=["game_id", "team", "pos"], how="left")
    p = p.merge(bk, on=["game_id", "team"], how="left")
    p[ABS] = p[ABS].fillna(0.0)
    return p


def pair_responses(p, a):
    """For each game, each player who played and each absent teammate: how much this player's minutes
    and 3PA rose above his own recent level in EARLIER games without that teammate (shrunk)."""
    q = p[["game_id", "team", "player_id", "game_date", "minutes", "fg3a", "min_ewm5", "min_ewm20",
           "fg3a_ewm5", "fg3a_ewm20"]].copy()
    q["r_min"] = q.minutes - (0.6 * q.min_ewm5 + 0.4 * q.min_ewm20)
    q["r_3pa"] = q.fg3a - (0.5 * q.fg3a_ewm5 + 0.5 * q.fg3a_ewm20)
    x = a[["game_id", "team", "absent_id"]].merge(q, on=["game_id", "team"]).dropna(subset=["r_min", "r_3pa"])
    x = x.sort_values(["player_id", "absent_id", "game_date"])
    g = x.groupby(["player_id", "absent_id"])
    x["n_prior"] = g.cumcount()
    x["resp_min"] = (g.r_min.cumsum() - x.r_min) / (x.n_prior + K_SHRINK)
    x["resp_3pa"] = (g.r_3pa.cumsum() - x.r_3pa) / (x.n_prior + K_SHRINK)
    return x[["game_id", "player_id", "absent_id", "resp_min", "resp_3pa", "n_prior"]]


def add_pairs(p, pr, a_used):
    """Sum pair responses over the absent teammates in a_used (actual or known-before-tip)."""
    use = pr.merge(a_used[["game_id", "absent_id"]], on=["game_id", "absent_id"])
    agg = use.groupby(["game_id", "player_id"]).agg(pair_min=("resp_min", "sum"), pair_3pa=("resp_3pa", "sum"),
                                                    pair_n=("n_prior", "sum")).reset_index()
    p = p.drop(columns=[c for c in PAIR if c in p], errors="ignore").merge(agg, on=["game_id", "player_id"], how="left")
    p[PAIR] = p[PAIR].fillna(0.0)
    return p


def fit(train, feats):
    return lgb.LGBMRegressor(objective="poisson", n_estimators=600, learning_rate=0.03, num_leaves=31,
                             min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             verbose=-1).fit(train[feats], train.fg3m)


if __name__ == "__main__":
    pm.SEASONS = [2023, 2024, 2025, 2026, 2027]
    p = pm.features()
    d = b1.load(pm.SEASONS)
    a, b = absent_table(d)
    pos = positions(pm.SEASONS)
    p = add_absence(p, a, b, pos)
    pr = pair_responses(p, a)
    p = add_pairs(p, pr, a)
    p.to_parquet("player_games_l2b.parquet", index=False)
    a.to_parquet("absences.parquet", index=False)
    pr.to_parquet("pair_responses.parquet", index=False)
    print("mean rotation minutes absent per team-game by season:",
          p.groupby("season").abs_min.mean().round(1).to_dict())

    train = p[p.season.isin([2024, 2025]) & (p.career_games >= 5)]
    val = p[(p.season == 2026) & (p.season_type == "regular") & (p.game_date < pm.VAL_END)]
    models = {"v1": pm.FEATS, "v2 team totals": FEATS2, "v3 + pairs": FEATS3}
    preds, rs, gb = {}, {}, {}
    for k, f in models.items():
        gb[k] = fit(train, f)
        rs[k] = pm.fit_r(np.clip(gb[k].predict(train[f]), .01, None), train.fg3m.values)
        val[k] = gb[k].predict(val[f])
    base = (val.min_ewm20 >= 10) & (val.career_games >= 20)
    sets = {"all regulars": base,
            "rotation teammate out (15+ min)": base & (val.abs_min > 0),
            "30+ rotation minutes out": base & (val.abs_min >= 30),
            "same-position rotation out": base & (val.abs_min_pos > 0),
            "no one out": base & (val.abs_min == 0)}
    rows = []
    for name, m in sets.items():
        for k in models:
            e = pm.evaluate(k, val.loc[m, k], val.loc[m, "fg3m"], rs[k])
            rows.append(dict(set=name, **{c: e[c] for c in ["model", "n", "MAE", "pois_dev", "mean_pred",
                                                          "mean_act", "LL_1+", "LL_2+", "LL_3+"]}))
    print("\nValidation, 2025-26 before 1 Feb (lower is better):")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    imp = pd.Series(gb["v3 + pairs"].booster_.feature_importance("gain"), FEATS3)
    print("\nAbsence + pair features' share of v3 weight:", (imp[ABS + PAIR] / imp.sum()).round(3).to_dict())
    big = val[base & (val.abs_min > 0)].copy()
    big["min_jump"] = big.minutes - big.min_ewm20
    print("\nWith a rotation teammate out, by how the player's minutes actually moved (diagnostic only):")
    print(big.groupby(pd.cut(big.min_jump, [-60, -5, 5, 10, 60]), observed=True).agg(
        n=("fg3m", "size"), actual_3pm=("fg3m", "mean"), v1=("v1", "mean"), v2=("v2 team totals", "mean"),
        v3=("v3 + pairs", "mean")).round(3).to_string())
