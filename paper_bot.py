"""
Beating The Bookie - paper-trading bot (forward test of the frozen v3 model).

  python3 paper_bot.py picks    evening (19:30 UTC = 20:30 Swiss): tonight's bet365 3PM ladders from the
                                MoviOdds board + latest NBA injury report -> frozen v3 -> paper picks
  python3 paper_bot.py grade    results, bet365 close and US-book fair close for earlier picks, then report
  python3 paper_bot.py report   summary only
  python3 paper_bot.py test     a picks run that writes to data/paper_test/ instead (never part of the test)

Frozen (models/v3_frozen.json): the model file, 1+ and 2+ rungs, bet when model probability >= 1/odds,
1 unit flat, known out = Out or Doubtful on the latest injury report. Nothing here is tuned during the test.

Files in data/paper/: picks.csv (one row per paper bet, graded in place), candidates_<YYYY-MM>.csv (every
price the model looked at), summary.txt, run_log.txt.
"""
import datetime as dt
import glob
import json
import os
import pathlib
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import lightgbm as lgb

import absence_model as am
import build_dataset as b1
import projection_model as pm
from layer2_vs_book import FULL, norm

OUT = pathlib.Path("data/paper")
OUT.mkdir(parents=True, exist_ok=True)
PICKS = OUT / "picks.csv"
META = json.load(open("models/v3_frozen.json"))
SEASONS = META["seasons_loaded"]
LIVE_SEASON = SEASONS[-1]
RUNGS = META["rule"]["rungs"]
ET = "America/New_York"
NOW = pd.Timestamp.now(tz="UTC")
BOX = b1.URL
SYN_BASE = 900_000_000
log = []


def say(msg):
    print(msg)
    log.append(str(msg))


# ---------------------------------------------------------------- inputs
def board():
    url = "https://moviodds.com/odds?" + urllib.parse.urlencode(dict(sport="basketball", groups="181449"))
    for _ in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "beat-ye-bookie/1.0", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            say(f"board error: {e}")
            time.sleep(10)
    return {}


def ladder_quotes(b):
    rows, games = [], []
    for m in b.get("matches", []):
        if (m.get("league") or "").strip().upper() != "NBA" or m.get("in_play"):
            continue
        ko = pd.Timestamp(m["kickoff_utc"])
        if ko <= NOW:
            continue
        games.append(dict(fixture_id=m["id"], away=m["away"], home=m["home"], kickoff_utc=m["kickoff_utc"]))
        for q in m.get("quotes", []):
            if str(q.get("group_id")) != "181449" or q.get("phase") == "inplay":
                continue
            f = [q.get("line"), q.get("label"), q.get("side"), q.get("handicap")]
            player = next((x for x in f if x and not str(x).replace(".", "").isdigit()), None)
            nums = [float(x) for x in f if x and str(x).replace(".", "").isdigit()]
            if not player or not nums:
                continue
            rows.append(dict(fixture_id=m["id"], game=f"{m['away']} @ {m['home']}", kickoff_utc=m["kickoff_utc"],
                             team_label=q.get("n2"), player=player, k=int(max(nums)), odds=float(q["odds_dec"]),
                             quote_ts=q.get("ts")))
    return pd.DataFrame(rows), pd.DataFrame(games)


def team_maps():
    t = pd.read_parquet(BOX.format(s=LIVE_SEASON), columns=["team_name", "team_abbreviation"]).drop_duplicates()
    nick = dict(zip(t.team_name, t.team_abbreviation))
    names = sorted(nick, key=len, reverse=True)
    def to_abbr(label):
        n = next((x for x in names if str(label).endswith(x)), None)
        return nick.get(n)
    return to_abbr


def injury_outs(slate, to_abbr):
    """Players Out/Doubtful for this slate: live NBA report if reachable, else the repo's change log."""
    try:
        import collect_injuries as ci
        from nbainjuries import injury
        t = NOW.tz_convert(ET).tz_localize(None).floor("15min")
        for _ in range(12):
            try:
                df = ci.clean(injury.get_reportdata(t.to_pydatetime(), return_df=True))
                src = f"live report {t:%Y-%m-%d %H:%M} ET"
                break
            except Exception:
                t -= pd.Timedelta(minutes=15)
        else:
            raise RuntimeError("no live report in the last 3 hours")
    except Exception as e:
        say(f"live injury report unavailable ({e}); using the repo's change log")
        files = sorted(glob.glob("data/injuries/status_changes_*.csv"))
        c = pd.concat([pd.read_csv(f) for f in files]) if files else pd.DataFrame()
        c = c[pd.to_datetime(c.report_utc).dt.tz_localize("UTC") <= NOW]
        df = c.sort_values("report_utc").groupby(["game_date", "team", "player"]).tail(1)
        src = f"change log to {c.report_utc.max()} UTC"
    df = df[df.game_date.astype(str) == str(slate)]
    out = df[df.status.isin(["Out", "Doubtful"])]
    return {(to_abbr(r.team), norm(r.player)) for r in out.itertuples()}, src


# ---------------------------------------------------------------- live features
def live_rows(games, outs, to_abbr):
    """Append one synthetic row per expected player for tonight's games, then run the training-time feature
    code unchanged. Synthetic rows have no stats, and every feature is shifted, so they only see the past."""
    d = b1.load(SEASONS)
    d["synthetic"] = False
    cur = d[(d.season == LIVE_SEASON) & d.played].sort_values("game_date")
    latest = cur.groupby("player_id").tail(1)
    syn = []
    for i, g in enumerate(games.itertuples()):
        slate = pd.Timestamp(g.kickoff_utc).tz_convert(ET).date()
        h, a = to_abbr(g.home), to_abbr(g.away)
        for team, opp, ha in ((h, a, "home"), (a, h, "away")):
            tg = cur[cur.team == team].game_id.unique()[-10:]
            members = latest[(latest.team == team) & latest.game_id.isin(tg)]
            for r in members.itertuples():
                if (team, norm(r.player)) in outs:
                    continue
                syn.append(dict(game_id=SYN_BASE + i, season=LIVE_SEASON, season_type="regular",
                                game_date=pd.Timestamp(slate), player_id=r.player_id, player=r.player, team=team,
                                opp=opp, home_away=ha, starter=False, played=True, reason=None, minutes=np.nan,
                                fg3m=np.nan, fg3a=np.nan, points=np.nan, rebounds=np.nan, assists=np.nan,
                                synthetic=True, fixture_id=g.fixture_id))
    s = pd.DataFrame(syn)
    da = pd.concat([d, s.drop(columns="fixture_id")], ignore_index=True).sort_values(["player_id", "game_date"])
    real_opp = b1.opponent_features

    def opp_live(dd):
        real = dd[~dd.synthetic.astype(bool)]
        of = real_opp(real)
        t = real[real.played].groupby(["season", "game_id", "opp"]).fg3a.sum().reset_index()
        lg, now_ = t.groupby("season").fg3a.mean(), t.groupby(["season", "opp"]).fg3a.mean()
        x = dd[dd.synthetic.astype(bool)][["season", "game_id", "opp"]].drop_duplicates()
        x["opp_fg3a_allowed_szn"] = [now_.get((se, o), np.nan) for se, o in zip(x.season, x.opp)]
        x["opp_fg3a_factor"] = x.opp_fg3a_allowed_szn / x.season.map(lg)
        return pd.concat([of, x[of.columns]])

    if not os.path.exists("game_lines.parquet"):           # 2026-27 has no historical lines: all NaN, as tested
        pd.DataFrame(columns=["game_id", "total", "home_fav_by", "home_implied_pts", "away_implied_pts"]).astype(
            float).to_parquet("game_lines.parquet")
    b1.load, b1.opponent_features = (lambda seasons=None: da.copy()), opp_live
    p = pm.features()
    a, back = am.absent_table(da)
    pos = am.positions(SEASONS)
    p = am.add_absence(p, a, back, pos)
    q = p[["game_id", "team", "player_id", "game_date", "minutes", "fg3a", "min_ewm5", "min_ewm20",
           "fg3a_ewm5", "fg3a_ewm20"]].copy()
    q["r_min"] = q.minutes - (0.6 * q.min_ewm5 + 0.4 * q.min_ewm20)
    q["r_3pa"] = q.fg3a - (0.5 * q.fg3a_ewm5 + 0.5 * q.fg3a_ewm20)
    synth = q.game_id >= SYN_BASE
    q.loc[synth, ["r_min", "r_3pa"]] = 0.0                   # tonight has no result: adds nothing to history
    x = a[["game_id", "team", "absent_id"]].merge(q, on=["game_id", "team"]).dropna(subset=["r_min", "r_3pa"])
    x = x.sort_values(["player_id", "absent_id", "game_date"])
    gx = x.groupby(["player_id", "absent_id"])
    x["n_prior"] = gx.cumcount()
    x["resp_min"] = (gx.r_min.cumsum() - x.r_min) / (x.n_prior + am.K_SHRINK)
    x["resp_3pa"] = (gx.r_3pa.cumsum() - x.r_3pa) / (x.n_prior + am.K_SHRINK)
    p = am.add_pairs(p, x[["game_id", "player_id", "absent_id", "resp_min", "resp_3pa", "n_prior"]], a)
    names = d[["player_id", "player"]].drop_duplicates("player_id").set_index("player_id").player
    gone = a[a.game_id >= SYN_BASE].assign(name=lambda z: z.absent_id.map(names))
    p = p[p.game_id >= SYN_BASE].merge(s[["game_id", "player_id", "fixture_id"]], on=["game_id", "player_id"])
    p["known_out"] = p.apply(lambda r: ", ".join(gone[(gone.game_id == r.game_id) & (gone.team == r.team)].name), axis=1)
    return p


# ---------------------------------------------------------------- modes
def picks():
    b = board()
    q, games = ladder_quotes(b)
    if games.empty or q.empty:
        say("no upcoming NBA games with ladder prices on the board")
        return
    done = set(pd.read_csv(PICKS, dtype=str).fixture_id) if PICKS.exists() else set()
    cand_done = set()
    for f in glob.glob(str(OUT / "candidates_*.csv")):
        cand_done |= set(pd.read_csv(f, dtype=str).fixture_id)
    games = games[~games.fixture_id.isin(done | cand_done)]
    if games.empty:
        say("tonight's games were already evaluated")
        return
    to_abbr = team_maps()
    games["slate"] = pd.to_datetime(games.kickoff_utc).dt.tz_convert(ET).dt.date
    today = NOW.tz_convert(ET).date()
    later = games[games.slate != today]
    if len(later):
        say(f"skipping {len(later)} games on later slates (evaluated on their own game day)")
    games = games[games.slate == today]
    if games.empty:
        say("no games on today's slate left to evaluate")
        return
    outs, src = set(), ""
    for slate in games.slate.unique():
        o, src = injury_outs(slate, to_abbr)
        outs |= o
    say(f"{NOW:%Y-%m-%d %H:%M} UTC | {len(games)} games, {len(q)} ladder prices | injuries: {src}, {len(outs)} out")
    rows = live_rows(games, outs, to_abbr)
    booster = lgb.Booster(model_file="models/v3_frozen.txt")
    rows["mu"] = np.clip(booster.predict(rows[META["features"]]), 0.02, None)
    P = pm.nb_ladder(rows.mu.values, META["nb_size_r"])
    rows["key"] = rows.player.map(norm)
    q = q[q.fixture_id.isin(games.fixture_id) & q.k.isin(RUNGS)].copy()
    q["team"] = q.team_label.map(to_abbr)
    q["key"] = q.player.map(norm)
    for i, k in enumerate(pm.LADDER):
        rows[f"p{k}"] = P[:, i]
    c = q.merge(rows[["fixture_id", "team", "key", "mu", "p1", "p2", "abs_min", "pair_min", "known_out"]],
                on=["fixture_id", "team", "key"], how="left")
    c["p_model"] = np.where(c.k == 1, c.p1, c.p2)
    c["p_implied"] = 1 / c.odds
    c["ev"] = c.p_model * c.odds - 1
    c["pick"] = c.p_model >= c.p_implied
    c["decision_utc"] = NOW.strftime("%Y-%m-%d %H:%M")
    c["slate"] = pd.to_datetime(c.kickoff_utc).dt.tz_convert(ET).dt.date.astype(str)
    c["quote_age_min"] = ((NOW - pd.to_datetime(c.quote_ts, format="ISO8601")).dt.total_seconds() / 60).round(0)
    c["injury_source"] = src
    c["model"] = META["name"]
    unmatched = c.p_model.isna().sum()
    c = c.drop(columns=["p1", "p2"])
    month = NOW.strftime("%Y-%m")
    cf = OUT / f"candidates_{month}.csv"
    c.to_csv(cf, mode="a", header=not cf.exists(), index=False)
    pk = c[c.pick].copy()
    for col in ["result_3pm", "hit", "void", "close_b365", "close_b365_age_min", "clv_b365", "p_fair_close",
                "fair_books", "clv_fair", "pl", "graded_utc"]:
        pk[col] = np.nan
    cols = ["decision_utc", "slate", "fixture_id", "game", "kickoff_utc", "team", "player", "k", "odds",
            "quote_ts", "quote_age_min", "mu", "p_model", "p_implied", "ev", "abs_min", "known_out",
            "injury_source", "model", "result_3pm", "hit", "void", "close_b365", "close_b365_age_min", "clv_b365",
            "p_fair_close", "fair_books", "clv_fair", "pl", "graded_utc"]
    pk[cols].to_csv(PICKS, mode="a", header=not PICKS.exists(), index=False)
    say(f"evaluated {c.p_model.notna().sum()} prices ({unmatched} unmatched to a model row); "
        f"{len(pk)} paper bets: {pk.k.value_counts().sort_index().to_dict()}")


def grade():
    if not PICKS.exists():
        say("no picks yet")
        return
    pk = pd.read_csv(PICKS)
    pk["graded_utc"] = pk.graded_utc.astype(object)
    todo = pk.graded_utc.isna()
    if not todo.any():
        return
    box = pd.read_parquet(BOX.format(s=LIVE_SEASON), columns=["athlete_display_name", "game_date", "team_abbreviation",
                                                               "three_point_field_goals_made", "minutes", "did_not_play"])
    box["key"] = box.athlete_display_name.map(norm)
    box["slate"] = pd.to_datetime(box.game_date).dt.strftime("%Y-%m-%d")
    played_teams = set(zip(box.slate, box.team_abbreviation))
    bx = box.set_index(["slate", "team_abbreviation", "key"])
    cl = pd.concat([pd.read_csv(f, low_memory=False) for f in glob.glob("data/bet365/*/threes_closing.csv")])
    cl = cl[cl.market == "ladder"].assign(key=lambda x: x.player.map(norm), k=lambda x: pd.to_numeric(x.threshold))
    cl["fixture_id"] = cl.fixture_id.astype(str)
    cl = cl.drop_duplicates(["fixture_id", "key", "k"], keep="last").set_index(["fixture_id", "key", "k"])
    fair = {}
    if os.path.exists("data/closing_props.csv"):
        pr = pd.read_csv("data/closing_props.csv")
        pr = pr[pr.market == "player_threes"]
        pr["slate"] = (pd.to_datetime(pr.commence_utc, utc=True).dt.tz_convert(ET)).dt.strftime("%Y-%m-%d")
        pr["team_h"] = pr.home.map(FULL)
        pr["key"] = pr.player.map(norm)
        w = pr.pivot_table(index=["slate", "team_h", "key", "book", "point"], columns="side",
                           values="price", aggfunc="last").dropna()
        w["p_over"] = (1 / w.Over) / (1 / w.Over + 1 / w.Under)
        fair = w.groupby(["slate", "key", "point"]).p_over.agg(["mean", "size"])
    for i in pk.index[todo]:
        r = pk.loc[i]
        if (r.slate, r.team) not in played_teams:
            continue                                         # box score not in yet
        try:
            b = bx.loc[(r.slate, r.team, norm(r.player))]
            b = b.iloc[0] if isinstance(b, pd.DataFrame) else b
            played = (not bool(b.did_not_play)) and (b.minutes or 0) > 0
        except KeyError:
            played = False
        ck = (str(r.fixture_id), norm(r.player), int(r.k))
        if ck in cl.index:
            c = cl.loc[ck]
            c = c.iloc[-1] if isinstance(c, pd.DataFrame) else c
            pk.loc[i, "close_b365"] = c.close_odds
            pk.loc[i, "close_b365_age_min"] = round((pd.Timestamp(r.kickoff_utc) - pd.Timestamp(c.close_ts)).total_seconds() / 60)
            pk.loc[i, "clv_b365"] = r.odds / c.close_odds - 1
        elif (NOW - pd.Timestamp(r.kickoff_utc)).total_seconds() < 36 * 3600:
            continue                                         # give the bet365 collector time
        fk = (r.slate, norm(r.player), r.k - 0.5)
        if len(fair) and fk in fair.index:
            pk.loc[i, "p_fair_close"] = fair.loc[fk, "mean"]
            pk.loc[i, "fair_books"] = fair.loc[fk, "size"]
            pk.loc[i, "clv_fair"] = r.odds * fair.loc[fk, "mean"] - 1
        if played:
            pk.loc[i, "result_3pm"] = b.three_point_field_goals_made
            pk.loc[i, "hit"] = float(b.three_point_field_goals_made >= r.k)
            pk.loc[i, "void"] = 0.0
            pk.loc[i, "pl"] = r.odds - 1 if b.three_point_field_goals_made >= r.k else -1.0
        else:
            pk.loc[i, ["void", "pl"]] = [1.0, 0.0]           # didn't play: stake returned
        pk.loc[i, "graded_utc"] = NOW.strftime("%Y-%m-%d %H:%M")
    pk.to_csv(PICKS, index=False)
    say(f"graded {(pk.graded_utc.notna() & todo).sum()} bets; {pk.graded_utc.isna().sum()} still open")


def report():
    if not PICKS.exists():
        return
    pk = pd.read_csv(PICKS)
    g = pk[pk.graded_utc.notna() & (pk.void == 0)]
    lines = [f"Paper test of {META['name']} (frozen {META['frozen_utc']} UTC) - report {NOW:%Y-%m-%d %H:%M} UTC",
             f"Bets placed {len(pk)}, graded {len(g)}, void {int((pk.void == 1).sum())}, open {int(pk.graded_utc.isna().sum())}"]

    def block(x, name):
        if x.empty:
            return f"{name}: no graded bets"
        f = x.dropna(subset=["clv_fair"])
        bclose = x.dropna(subset=["clv_b365"])
        return (f"{name}: {len(x)} bets | P/L {x.pl.sum():+.2f}u, ROI {100 * x.pl.mean():+.1f}% "
                f"(±{100 * x.pl.std() / max(np.sqrt(len(x)), 1):.1f}) | hit {x.hit.mean():.3f} vs implied {x.p_implied.mean():.3f} | "
                f"fair CLV {100 * f.clv_fair.mean():+.1f}% on {len(f)} ({(f.clv_fair > 0).mean():.0%} beat) | "
                f"bet365 CLV {100 * bclose.clv_b365.mean():+.1f}% on {len(bclose)} ({(bclose.clv_b365 > 0).mean():.0%} beat)")
    lines.append(block(g, "All"))
    for k in RUNGS:
        lines.append(block(g[g.k == k], f"{k}+"))
    lines.append("Pass test (about 300 bets): fair CLV above 0 on average and more than half of bets beating the close.")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    say("\n".join(lines))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"
    if mode == "test":
        OUT = pathlib.Path("data/paper_test")
        OUT.mkdir(parents=True, exist_ok=True)
        PICKS = OUT / "picks.csv"
    try:
        if mode == "test":
            picks()
        elif mode == "picks":
            picks()
            grade()
        elif mode == "grade":
            grade()
        report()
    finally:
        with open(OUT / "run_log.txt", "a") as f:
            f.write("\n".join([f"--- {mode} {NOW:%Y-%m-%d %H:%M} UTC"] + log) + "\n")
