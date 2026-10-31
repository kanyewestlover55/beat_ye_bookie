"""
Beating The Bookie - news test: injury reports vs bet365 3PM ladder prices (2026-27).

For each game in the bet365 feed:
  - morning price M = last feed price at or before 10:00 UTC on slate day; close = last before tip
  - each rotation player's availability at M and at tip, from the official injury report
    (Out 0, Doubtful .25, Questionable .5, Probable .9, Available / removed / not listed 1)
  - "late news" = a rotation player (20+ min per game recently) whose availability changed by 0.5+
    between M and tip. Late OUT = teammates should shorten; late IN = teammates should drift.
Then: (1) how big and how common the teammates' moves are with and without late news,
(2) how fast bet365 moved after the report (feed updates about hourly, so resolution is ~1 h),
(3) whether prices moved before the report (news reaching the book first),
(4) when late news breaks, in hours before tip and Swiss clock time.
"""
import numpy as np
import pandas as pd

import build_dataset as b1
from layer2_vs_book import norm

AVAIL = {"Out": 0.0, "Doubtful": 0.25, "Questionable": 0.5, "Probable": 0.9, "Available": 1.0, "Removed": 1.0}
ROTATION_MIN = 20
BOX = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_nba_player_boxscores/player_box_{}.parquet"


def nick_of(name, nicks):
    return next((n for n in nicks if str(name).endswith(n)), None)


def load_box():
    raw = pd.concat([pd.read_parquet(BOX.format(s), columns=["athlete_display_name", "team_name", "game_date",
                                                                "minutes", "did_not_play"])
                     for s in (2026, 2027)])
    raw["key"] = raw.athlete_display_name.map(norm)
    raw["game_date"] = pd.to_datetime(raw.game_date)
    raw["minutes"] = pd.to_numeric(raw.minutes, errors="coerce")
    played = raw[(raw.did_not_play != True) & (raw.minutes > 0)].sort_values(["key", "game_date"]).copy()
    played["min_ewm"] = played.groupby("key").minutes.transform(lambda s: s.ewm(halflife=10).mean())
    return raw.sort_values("game_date"), played[["key", "game_date", "min_ewm"]].sort_values("game_date")


def asof(df, right, by, on="date"):
    """Latest row of `right` strictly before each row of df (per `by`)."""
    l = df.sort_values(on).reset_index()
    r = right.rename(columns={"game_date": on}).sort_values(on)
    m = pd.merge_asof(l, r, on=on, by=by, allow_exact_matches=False)
    return m.set_index("index").reindex(df.index)


def main():
    raw, played = load_box()
    nicks = sorted(raw.team_name.dropna().unique(), key=len, reverse=True)

    # ---------- feed prices ----------
    h = pd.read_csv("data/bet365/2026-10/threes_history.csv", low_memory=False)
    h = h[(h.market == "ladder") & (h.suspended != True)].copy()
    h["ts"] = pd.to_datetime(h.ts, format="ISO8601", utc=True)
    h["kick"] = pd.to_datetime(h.kickoff_utc, utc=True)
    h = h[h.ts <= h.kick]
    h["k"] = pd.to_numeric(h.threshold)
    h["q"] = 1 / h.odds_dec
    h["key"] = h.player.map(norm)
    h["date"] = pd.to_datetime(h.kick.dt.tz_convert("America/New_York").dt.date)
    h["M"] = (h.kick - pd.Timedelta(hours=10)).dt.normalize() + pd.Timedelta(hours=10)
    teams = h.game.str.split(" @ ", expand=True)
    h["away"], h["home"] = teams[0].map(lambda s: nick_of(s, nicks)), teams[1].map(lambda s: nick_of(s, nicks))
    roster = raw[["key", "game_date", "team_name"]].dropna()
    team_now = asof(h[["key", "date"]].drop_duplicates(), roster, "key")
    h = h.merge(pd.concat([h[["key", "date"]].drop_duplicates(), team_now.team_name], axis=1),
                on=["key", "date"], how="left")
    h = h[(h.team_name == h.home) | (h.team_name == h.away)]

    # ---------- injury timelines ----------
    inj = pd.read_csv("data/injuries/status_changes_2026-27.csv")
    inj["t"] = pd.to_datetime(inj.report_utc).dt.tz_localize("UTC")
    inj["date"] = pd.to_datetime(inj.game_date)
    inj["team_name"] = inj.team.map(lambda s: nick_of(s, nicks))
    inj["key"] = inj.player.map(norm)
    inj["avail"] = inj.status.map(AVAIL)
    games = h.groupby("fixture_id").agg(date=("date", "first"), home=("home", "first"), away=("away", "first"),
                                        kick=("kick", "first"), M=("M", "first")).reset_index()
    gl = pd.concat([games.assign(team_name=games.home), games.assign(team_name=games.away)])
    inj = inj.merge(gl[["fixture_id", "date", "team_name", "kick", "M"]], on=["date", "team_name"])

    def avail_at(g, t):
        x = g[g.t <= t]
        return 1.0 if x.empty else x.avail.iloc[-1]

    rows = []
    for (fid, team, key), g in inj.sort_values("t").groupby(["fixture_id", "team_name", "key"]):
        M, kick = g.M.iloc[0], g.kick.iloc[0]
        aM, aT = avail_at(g, M), avail_at(g, kick)
        late = g[(g.t > M) & (g.t <= kick)]
        t_change = late.t.iloc[(late.avail - aM).abs().values.argmax()] if len(late) else pd.NaT
        rows.append(dict(fixture_id=fid, team_name=team, key=key, player=g.player.iloc[0], aM=aM, aT=aT,
                         d=aT - aM, t_event=t_change, kick=kick, M=M, date=g.date.iloc[0]))
    ev = pd.DataFrame(rows)
    imp = asof(ev[["key", "date"]], played, "key")
    ev["min_ewm"] = imp.min_ewm.values
    ev["rotation"] = ev.min_ewm >= ROTATION_MIN
    rot = ev[ev.rotation]
    print(f"Games in feed: {h.fixture_id.nunique()} | rotation players on an injury report: {len(rot)}")
    print(f"  ended the day Out/Doubtful: {(rot.aT <= .25).sum()}, of which already Out/Doubtful by 10:00 UTC: "
          f"{((rot.aT <= .25) & (rot.aM <= .25)).sum()}")
    late = rot[rot.d.abs() >= 0.5].copy()
    late["kind"] = np.where(late.d < 0, "late OUT", "late IN")
    print(f"  late news (availability changed 0.5+ between 10:00 UTC and tip): {len(late)} "
          f"({(late.kind == 'late OUT').sum()} downgrades, {(late.kind == 'late IN').sum()} upgrades)")

    # ---------- (4) when late news breaks ----------
    late["h_before_tip"] = (late.kick - late.t_event).dt.total_seconds() / 3600
    late["swiss"] = late.t_event.dt.tz_convert("Europe/Zurich").dt.strftime("%H:%M")
    print("\nWhen late news breaks (hours before tip):",
          pd.cut(late.h_before_tip, [0, 1, 2, 4, 8, 24]).value_counts().sort_index().to_dict())
    print("Swiss clock time of late news:", sorted(late.swiss.tolist()))

    # ---------- (1) teammates' morning -> close moves ----------
    hs = h.sort_values("ts")
    pk = hs.groupby(["fixture_id", "team_name", "key", "k"])
    px = pk.apply(lambda g: pd.Series(dict(qM=g[g.ts <= g.M.iloc[0]].q.iloc[-1] if (g.ts <= g.M.iloc[0]).any()
                                           else np.nan, qC=g.q.iloc[-1]))).reset_index().dropna()
    px["dq"] = px.qC - px.qM
    news_team = late.groupby(["fixture_id", "team_name"]).agg(
        out_min=("min_ewm", lambda s: s[late.loc[s.index, "d"] < 0].sum()),
        in_min=("min_ewm", lambda s: s[late.loc[s.index, "d"] > 0].sum()),
        injured=("key", set)).reset_index()
    px = px.merge(news_team, on=["fixture_id", "team_name"], how="left")
    px = px[~px.apply(lambda r: isinstance(r.injured, set) and r.key in r.injured, axis=1)]   # teammates only
    px["group"] = np.select([px.out_min.fillna(0) > 0, px.in_min.fillna(0) > 0], ["late OUT", "late IN"], "no late news")
    summ = px.groupby("group").agg(prices=("dq", "size"), teams=("team_name", lambda s: s.index.size),
                                   mean_move_pts=("dq", lambda s: 100 * s.mean()),
                                   abs_move_pts=("dq", lambda s: 100 * s.abs().mean()),
                                   shortened=("dq", lambda s: (s > .001).mean()),
                                   drifted=("dq", lambda s: (s < -.001).mean()))
    summ["teams"] = px.groupby("group").apply(lambda g: g[["fixture_id", "team_name"]].drop_duplicates().shape[0])
    summ["share_of_all_movement"] = px.groupby("group").dq.apply(lambda s: s.abs().sum()) / px.dq.abs().sum()
    summ["share_of_prices"] = summ.prices / summ.prices.sum()
    print("\nTeammates' ladder moves, morning (10:00 UTC) to close, by what happened to a rotation teammate:")
    print(summ.round(3).to_string())
    for k in (1, 2):
        s = px[px.k == k].groupby("group").dq.agg(["size", lambda x: 100 * x.mean()])
        s.columns = ["prices", "mean_move_pts"]
        print(f"  {k}+ only:", s.round(2).to_dict("index"))

    # ---------- (2)+(3) reaction speed around the report ----------
    out = []
    for r in late.itertuples():
        tm = hs[(hs.fixture_id == r.fixture_id) & (hs.team_name == r.team_name) & (hs.key != r.key)]
        if tm.empty:
            continue
        snaps = np.sort(tm.ts.unique())
        before = snaps[snaps <= r.t_event]
        after = snaps[snaps > r.t_event]
        if len(before) == 0 or len(after) == 0:
            continue
        t_pre, t_post = before[-1], after[0]
        t_pre2 = before[before <= r.t_event - pd.Timedelta(hours=2)]
        def q_at(t):
            return tm[tm.ts <= t].groupby(["key", "k"]).q.last()
        qpre, qpost, qclose = q_at(t_pre), q_at(t_post), q_at(r.kick)
        qold = q_at(t_pre2[-1]) if len(t_pre2) else None
        sign = -1 if r.d < 0 else 1                    # late OUT: teammates should shorten (+q)
        j = qpre.index.intersection(qpost.index).intersection(qclose.index)
        total, first = (qclose[j] - qpre[j]).mean(), (qpost[j] - qpre[j]).mean()
        out.append(dict(player=r.player, kind=r.kind, min_ewm=r.min_ewm, lag_min=(t_post - r.t_event) / pd.Timedelta(minutes=1),
                        pre_to_close_pts=-sign * 100 * total, pre_to_first_pts=-sign * 100 * first,
                        before_report_pts=(-sign * 100 * (qpre[j] - qold.reindex(j)).mean()) if qold is not None else np.nan,
                        h_before_tip=r.h_before_tip))
    rx = pd.DataFrame(out)
    print("\nReaction around each late news report (teammates' average move, in the expected direction, pts):")
    print(rx.round(2).sort_values("lag_min").to_string(index=False))
    if len(rx):
        print(f"\nAverage: 2h before report -> report {rx.before_report_pts.mean():+.2f} pts | report -> first feed update "
              f"{rx.pre_to_first_pts.mean():+.2f} pts (median lag {rx.lag_min.median():.0f} min) | report -> close "
              f"{rx.pre_to_close_pts.mean():+.2f} pts")
    late.to_csv("news_late_events.csv", index=False)
    px.to_csv("news_teammate_moves.csv", index=False)
    rx.to_csv("news_reaction.csv", index=False)


if __name__ == "__main__":
    main()
