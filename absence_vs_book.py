"""
Beating The Bookie - absence model vs bet365 (2026-27 games).

v1 (no absence info) and v3 (absences + pair responses) are fitted on 2023-24, 2024-25 and 2025-26
before 1 Feb. For 2026-27, v3 only gets absences that were already Out/Doubtful on the injury report
at 10:00 UTC on game day: what you could know when betting morning prices. Then the paper strategy
from 29 Oct (bet the decision-time ladder price when the model is 5+ points above the margin-adjusted morning
price), on the agreed paper book: 1+ and 2+ rungs only.
"""
import numpy as np
import pandas as pd

import absence_model as am
import build_dataset as b1
import projection_model as pm
from layer2_vs_book import ABBR, norm

AVAIL = {"Out": 0.0, "Doubtful": 0.25, "Questionable": 0.5, "Probable": 0.9, "Available": 1.0, "Removed": 1.0}


def known_at(a, d, hour_utc):
    """Keep absences listed Out/Doubtful on the injury report at hour_utc (UTC) on game day (ET date)."""
    inj = pd.read_csv("data/injuries/status_changes_2026-27.csv")
    inj["t"] = pd.to_datetime(inj.report_utc)
    inj["key"] = inj.player.map(norm)
    inj["M"] = pd.to_datetime(inj.game_date) + pd.Timedelta(hours=hour_utc)
    st = inj[inj.t <= inj.M].sort_values("t").groupby(["game_date", "key"]).status.last().map(AVAIL).reset_index()
    ids = d[["game_id", "game_date", "player_id", "player"]].drop_duplicates(["game_id", "player_id"])
    ids["key"] = ids.player.map(norm)
    ids["game_date"] = ids.game_date.dt.strftime("%Y-%m-%d")
    # absent players may have no box row that day: map names to ids from any game
    name_id = ids.drop_duplicates("key").set_index("key").player_id
    st2 = st[st.status <= 0.25].assign(player_id=lambda x: x.key.map(name_id)).dropna(subset=["player_id"])
    games = d[["game_id", "game_date"]].drop_duplicates().assign(game_date=lambda x: x.game_date.dt.strftime("%Y-%m-%d"))
    known = st2.merge(games, on="game_date")[["game_id", "player_id"]].drop_duplicates()
    return a.merge(known.rename(columns={"player_id": "absent_id"}), on=["game_id", "absent_id"])


def feed_prices(hour_utc):
    h = pd.read_csv("data/bet365/2026-10/threes_history.csv", low_memory=False)
    h = h[(h.market == "ladder") & (h.suspended != True)].copy()
    h["ts"] = pd.to_datetime(h.ts, format="ISO8601", utc=True)
    h["kick"] = pd.to_datetime(h.kickoff_utc, utc=True)
    h = h[h.ts <= h.kick]
    h["k"] = pd.to_numeric(h.threshold)
    h = h[h.k.isin([1, 2])]
    h["M"] = (h.kick - pd.Timedelta(hours=10)).dt.normalize() + pd.Timedelta(hours=hour_utc)
    h = h.sort_values("ts")
    mo = h[h.ts <= h.M].groupby(["fixture_id", "game", "kickoff_utc", "player", "k"]).odds_dec.last().rename("mo")
    cl = h.groupby(["fixture_id", "game", "kickoff_utc", "player", "k"]).odds_dec.last().rename("co")
    f = pd.concat([mo, cl], axis=1).dropna().reset_index()
    f["key"] = f.player.map(norm)
    f["home_abbr"] = f.game.str.split(" @ ").str[1].str.split().str[0].replace(ABBR)
    f["slate"] = (pd.to_datetime(f.kickoff_utc, utc=True) - pd.Timedelta(hours=10)).dt.date
    return f


def main():
    pm.SEASONS = [2023, 2024, 2025, 2026, 2027]
    p = pd.read_parquet("player_games_l2b.parquet")
    d = b1.load(pm.SEASONS)
    a, b = am.absent_table(d)
    pos = am.positions(pm.SEASONS)
    pr = pd.read_parquet("pair_responses.parquet")

    train = p[(p.season.isin([2024, 2025]) | ((p.season == 2026) & (p.game_date < pm.VAL_END)))
              & (p.career_games >= 5)]
    g1, g3 = am.fit(train, pm.FEATS), am.fit(train, am.FEATS3)
    r1 = pm.fit_r(np.clip(g1.predict(train[pm.FEATS]), .01, None), train.fg3m.values)
    r3 = pm.fit_r(np.clip(g3.predict(train[am.FEATS3]), .01, None), train.fg3m.values)

    out = []
    for label, hour_utc in (("10:00 UTC (11:00 Swiss)", 10), ("19:30 UTC (20:30 Swiss)", 19.5)):
        live = p[p.season == 2027].copy()
        a27 = a[a.game_id.isin(live.game_id)]
        ak = known_at(a27, d[d.season == 2027], hour_utc)
        print(f"\n=========== decision time {label} ===========")
        print(f"2026-27 rotation absences: {len(a27)}, listed Out/Doubtful by then: {len(ak)}")
        live = am.add_absence(live, ak, b, pos)
        live = am.add_pairs(live, pr, ak)
        live["v1"], live["v3"] = g1.predict(live[pm.FEATS]), g3.predict(live[am.FEATS3])
        live["key"] = live.player.map(norm)
        live["slate"] = pd.to_datetime(live.game_date).dt.date
        live["home_abbr"] = np.where(live.home == 1, live.team, live.opp)

        f = feed_prices(hour_utc).merge(live[["slate", "home_abbr", "key", "fg3m", "v1", "v3", "abs_min", "pair_min"]],
                                on=["slate", "home_abbr", "key"])
        f["hit"] = (f.fg3m >= f.k).astype(float)
        f["qm"], f["qc"] = 1 / f.mo, 1 / f.co
        for v, r in (("v1", r1), ("v3", r3)):
            P = pm.nb_ladder(f[v].values, r)
            f[f"p_{v}"] = np.where(f.k == 1, P[:, 0], P[:, 1])
        f["qm_adj"] = f.qm - f.groupby("k").qm.transform("mean") + f.groupby("k").hit.transform("mean")
        f["affected"] = f.abs_min > 0
        print(f"Matched prices (1+ and 2+): {len(f)}; with a known rotation teammate out: {f.affected.sum()}")

        def ll(p, y):
            p = np.clip(p, 1e-4, 1 - 1e-4)
            return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        rows = []
        for name, m in (("all", f.index == f.index), ("teammate known out", f.affected), ("no one out", ~f.affected)):
            g = f[m]
            rows.append(dict(set=name, n=len(g), hit=g.hit.mean(), LL_v1=ll(g.p_v1, g.hit), LL_v3=ll(g.p_v3, g.hit),
                             LL_book_then=ll(g.qm, g.hit), LL_close=ll(g.qc, g.hit)))
        print("\nLog loss on 1+ and 2+ (book prices include the margin):")
        print(pd.DataFrame(rows).round(4).to_string(index=False))

        print("\nPaper strategy: bet the decision-time price when the model is 5+ points above it (margin-adjusted)")
        for v in ("v1", "v3"):
            for name, m in (("all", f.index == f.index), ("teammate known out", f.affected)):
                s = f[m & (f[f"p_{v}"] - f.qm_adj > 0.05)]
                if len(s) == 0:
                    continue
                roi = s.hit * s.mo - 1
                print(f"  {v:3s} {name:20s} bets {len(s):4d} | hit {s.hit.mean():.3f} vs implied {s.qm.mean():.3f} | "
                      f"ROI {100 * roi.mean():+.1f}% (±{100 * roi.std() / np.sqrt(len(s)):.1f}) | "
                      f"CLV vs close {100 * (s.mo / s.co - 1).mean():+.1f}%")
        out.append(f.assign(decision=label))
    pd.concat(out).to_csv("absence_vs_bet365.csv", index=False)


if __name__ == "__main__":
    main()
