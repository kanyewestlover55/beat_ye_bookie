"""
Beating The Bookie - projected starters (research for v4; does not touch the frozen v3 or the paper bot).

For every team-game, project the starting five before tip:
  1. start from the previous game's starters and drop those who are out
  2. fill each gap with the player who started the last time that player sat (history rule)
  3. otherwise, the available bench player at the same position (G/F/C) with the most recent minutes
"Out" here = didn't play (perfect injury information, an upper bound). Part B adds the projection to the
3PM model as features and checks it on the usual validation split (2025-26 before 1 Feb; holdout untouched).
"""
import numpy as np
import pandas as pd

import absence_model as am
import build_dataset as b1
import projection_model as pm

SEASONS = [2023, 2024, 2025, 2026, 2027]


def project(d, pos):
    p = d[d.played].sort_values(["player_id", "game_date"]).copy()
    p["m_pre"] = p.groupby("player_id").minutes.transform(lambda s: s.shift(1).ewm(halflife=10).mean())
    p = p.merge(pos, left_on="player_id", right_index=True, how="left")
    rows = []
    for (season, team), tp in p.groupby(["season", "team"]):
        games = tp[["game_id", "game_date"]].drop_duplicates().sort_values("game_date").game_id.tolist()
        by_g = {g: x for g, x in tp.groupby("game_id")}
        replaced_by = {}                                  # player -> who started last time he sat
        prev_start = None
        for g in games:
            x = by_g[g]
            playing = set(x.player_id)
            actual = set(x[x.starter].player_id)
            if prev_start is not None and len(actual) == 5:
                outs = [s for s in prev_start if s not in playing]
                keep = [s for s in prev_start if s in playing]
                lineup_hist, lineup_pos = list(keep), list(keep)
                for o in outs:
                    # history rule
                    cand = [c for c in replaced_by.get(o, []) if c in playing and c not in lineup_hist]
                    # position rule (also the fallback)
                    o_pos = pos.get(o, "F")
                    bench = x[~x.player_id.isin(lineup_pos)].sort_values("m_pre", ascending=False)
                    same = bench[bench.pos == o_pos]
                    pick_pos = (same if len(same) else bench).player_id.iloc[0] if len(bench) else None
                    if pick_pos is not None:
                        lineup_pos.append(pick_pos)
                    if cand:
                        lineup_hist.append(cand[0])
                    else:
                        bench_h = x[~x.player_id.isin(lineup_hist)].sort_values("m_pre", ascending=False)
                        same_h = bench_h[bench_h.pos == o_pos]
                        if len(bench_h):
                            lineup_hist.append((same_h if len(same_h) else bench_h).player_id.iloc[0])
                rows.append(dict(game_id=g, team=team, season=season, n_out=len(outs),
                                 hit_naive=len(set(keep) & actual), hit_pos=len(set(lineup_pos) & actual),
                                 hit_hist=len(set(lineup_hist) & actual), proj=lineup_hist, actual=actual))
                # learn: who stepped in for each missing previous starter
                newcomers = [a for a in actual if a not in prev_start]
                for o in outs:
                    if newcomers:
                        replaced_by[o] = newcomers
            if len(actual) == 5:
                prev_start = actual
    return pd.DataFrame(rows)


if __name__ == "__main__" and __import__("sys").argv[-1] != "b":
    d = b1.load(SEASONS)
    pos = am.positions(SEASONS)
    r = project(d[d.season_type == "regular"], pos)
    r.to_pickle("projected_starters.pkl")
    ev = r[r.season.isin([2024, 2025, 2026])]
    print(f"Team-games: {len(ev)}; with at least one previous starter out: {(ev.n_out > 0).sum()} "
          f"({(ev.n_out > 0).mean():.0%})")
    for name, m in (("all team-games", ev.index == ev.index), ("a previous starter out", ev.n_out > 0),
                    ("2+ previous starters out", ev.n_out >= 2)):
        g = ev[m]
        filled = g.n_out.sum()
        print(f"\n{name} ({len(g)}):")
        print(f"  starters right out of 5: naive {g.hit_naive.mean():.2f} | position rule {g.hit_pos.mean():.2f} | "
              f"history rule {g.hit_hist.mean():.2f}")
        print(f"  all five right: position {(g.hit_pos == 5).mean():.0%} | history {(g.hit_hist == 5).mean():.0%}")
        if filled:
            print(f"  gaps filled correctly: position {(g.hit_pos - g.hit_naive).sum() / filled:.0%} | "
                  f"history {(g.hit_hist - g.hit_naive).sum() / filled:.0%}")


def part_b():
    """Add proj_start / promoted to v3's features and compare on the validation split."""
    r = pd.read_pickle("projected_starters.pkl")
    ps = r[["game_id", "team", "proj"]].explode("proj").rename(columns={"proj": "player_id"})
    ps["proj_start"] = 1.0
    p = pd.read_parquet("player_games_l2b.parquet")
    covered = set(r.game_id)
    p = p.merge(ps[["game_id", "player_id", "proj_start"]].astype({"player_id": p.player_id.dtype}),
                on=["game_id", "player_id"], how="left")
    p.loc[p.game_id.isin(covered) & p.proj_start.isna(), "proj_start"] = 0.0
    p["promoted"] = np.where(p.proj_start.notna(), ((p.proj_start == 1) & (p.last_start == 0)).astype(float), np.nan)
    feats4 = am.FEATS3 + ["proj_start", "promoted"]
    train = p[p.season.isin([2024, 2025]) & (p.career_games >= 5)]
    val = p[(p.season == 2026) & (p.season_type == "regular") & (p.game_date < pm.VAL_END)].copy()
    rs = {}
    for k, f in (("v3", am.FEATS3), ("v4", feats4)):
        g = am.fit(train, f)
        rs[k] = pm.fit_r(np.clip(g.predict(train[f]), .01, None), train.fg3m.values)
        val[k] = g.predict(val[f])
    base = (val.min_ewm20 >= 10) & (val.career_games >= 20)
    sets = {"all regulars": base, "projected promoted to starter": base & (val.promoted == 1),
            "starter last game, projected bench": base & (val.last_start == 1) & (val.proj_start == 0),
            "rotation teammate out": base & (val.abs_min > 0)}
    rows = []
    for name, m in sets.items():
        for k in ("v3", "v4"):
            e = pm.evaluate(k, val.loc[m, k], val.loc[m, "fg3m"], rs[k])
            rows.append(dict(set=name, **{c: e[c] for c in ["model", "n", "MAE", "pois_dev", "mean_pred",
                                                          "mean_act", "LL_1+", "LL_2+", "LL_3+"]}))
    print("\nPart B - validation, 2025-26 before 1 Feb (lower is better):")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    pr = val[base & (val.promoted == 1)]
    print(f"\nPromoted players: minutes {pr.minutes.mean():.1f} vs recent {pr.min_ewm20.mean():.1f}; actually started "
          f"{pr.starter.mean():.0%}; 3PM actual {pr.fg3m.mean():.2f}, v3 {pr.v3.mean():.2f}, v4 {pr.v4.mean():.2f}")
    big = val[base & ((val.minutes - val.min_ewm20) >= 10)]
    print(f"Minutes jumped 10+: n {len(big)}, 3PM actual {big.fg3m.mean():.2f}, v3 {big.v3.mean():.2f}, v4 {big.v4.mean():.2f}; "
          f"share flagged as promoted {(big.promoted == 1).mean():.0%}")


if __name__ == "__main__" and __import__("sys").argv[-1] == "b":
    part_b()
