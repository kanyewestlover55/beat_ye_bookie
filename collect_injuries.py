"""
Beating The Bookie - official NBA injury report collector.

The NBA publishes its injury report as PDFs (every 15 minutes since 22 Dec 2025, hourly before)
and keeps them online, so this collector never has to be on time: each run processes every
report it hasn't seen yet. It stores only CHANGES, one row each time a player is added to a
game's report, changes status, or is removed:

  data/injuries/status_changes_<season>.csv   report time (ET and UTC), game, team, player,
                                              status, previous status, reason, change type
  data/injuries/reports_log.csv               every report time tried: ok / missing / error

Modes
  live (default, scheduled): from the last processed report up to 20 minutes ago, every 15 min.
  backfill: --start/--end dates (ET), --step minutes. Resumes where it stopped; run the same
            range again until it reports "complete". Run date ranges in chronological order.
Needs Java (tabula) and `pip install nbainjuries pandas`.
"""
import argparse
import json
import socket
import os
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

ET, UTC = ZoneInfo("America/New_York"), ZoneInfo("UTC")
DATA = "data/injuries"
LOG = f"{DATA}/reports_log.csv"
FMT_GAP = (datetime(2025, 12, 19, 15, 30), datetime(2025, 12, 19, 16, 45))   # no valid URLs between
NEW_15MIN = datetime(2025, 12, 22, 9, 0)
COLS = ["report_et", "report_utc", "game_date", "game_time_et", "matchup", "team", "player",
        "status", "prev_status", "reason", "change"]


def season_of(d):
    y = d.year if d.month >= 8 else d.year - 1
    return f"{y}-{(y + 1) % 100:02d}"


def player_name(s):
    s = " ".join(str(s).split())
    if "," in s:                                   # "Brown, Jaylen" -> "Jaylen Brown"
        last, first = s.split(",", 1)
        s = f"{first.strip()} {last.strip()}"
    return s


def load_log():
    if os.path.exists(LOG):
        return pd.read_csv(LOG, dtype=str)
    return pd.DataFrame(columns=["report_et", "mode", "result", "rows", "note"])


def load_state(mode):
    path = f"{DATA}/state_{mode}.json"
    return json.load(open(path)) if os.path.exists(path) else {}


def clean(df):
    df = df.rename(columns=lambda c: str(c).strip())
    for c in ["Game Date", "Game Time", "Matchup", "Team"]:
        df[c] = df[c].replace("", pd.NA).ffill()
    df = df[df["Player Name"].notna() & (df["Player Name"].astype(str).str.strip() != "")].copy()
    df["player"] = df["Player Name"].map(player_name)
    df["game_date"] = pd.to_datetime(df["Game Date"], format="%m/%d/%Y").dt.date.astype(str)
    df["game_time_et"] = df["Game Time"].astype(str).str.replace("(ET)", "", regex=False).str.strip()
    return df.rename(columns={"Matchup": "matchup", "Team": "team", "Current Status": "status",
                              "Reason": "reason"})


def diff(df, state, ts):
    """Compare one report with the last known statuses; return change rows and update state."""
    rows, seen = [], set()
    base = dict(report_et=ts.strftime("%Y-%m-%d %H:%M"),
                report_utc=ts.replace(tzinfo=ET).astimezone(UTC).strftime("%Y-%m-%d %H:%M"))
    for r in df.itertuples(index=False):
        key = f"{r.game_date}|{r.matchup}|{r.player}"
        seen.add(key)
        prev = state.get(key)
        if prev is None or prev["status"] != r.status:
            rows.append(dict(base, game_date=r.game_date, game_time_et=r.game_time_et, matchup=r.matchup,
                             team=r.team, player=r.player, status=r.status,
                             prev_status=prev["status"] if prev else "", reason=r.reason,
                             change="new" if prev is None else "status"))
        state[key] = dict(status=r.status, team=r.team, game_time_et=r.game_time_et, reason=r.reason)
    games = {(g, m) for g, m in zip(df.game_date, df.matchup)}
    for key in list(state):                                     # dropped off a game still listed
        g, m, p = key.split("|", 2)
        if (g, m) in games and key not in seen:
            s = state.pop(key)
            rows.append(dict(base, game_date=g, game_time_et=s["game_time_et"], matchup=m, team=s["team"],
                             player=p, status="Removed", prev_status=s["status"], reason=s["reason"],
                             change="removed"))
    cutoff = str((ts - timedelta(days=2)).date())
    for key in [k for k in state if k.split("|", 1)[0] < cutoff]:
        state.pop(key)
    return rows


def timestamps(start, end, step):
    """Report slots: hourly before 22 Dec 2025, then every `step` minutes (15 or 60)."""
    t = start
    while t <= end:
        if t < NEW_15MIN and t.minute:
            t = t.replace(minute=0) + timedelta(hours=1)
            continue
        if not (FMT_GAP[0] < t < FMT_GAP[1]):
            yield t
        t += timedelta(minutes=step if t >= NEW_15MIN else 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="live", choices=["live", "backfill"])
    ap.add_argument("--start"), ap.add_argument("--end")
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--max-minutes", type=float, default=None)
    a = ap.parse_args()
    from nbainjuries import injury
    from nbainjuries._exceptions import URLRetrievalError

    socket.setdefaulttimeout(60)               # nbainjuries makes requests without a timeout
    os.makedirs(DATA, exist_ok=True)
    log, state = load_log(), load_state(a.mode)
    now = datetime.now(ET).replace(tzinfo=None)
    # done = read OK, or missing and final. Errors are always retried. In live mode a missing
    # report is retried for 24 hours, in case the NBA uploads it late.
    final_missing = log.result == "missing"
    if a.mode == "live":
        final_missing &= pd.to_datetime(log.report_et) < now - timedelta(hours=24)
    done = set(log[(log.result == "ok") | final_missing].report_et)
    if a.mode == "live":                       # look back 3 days; already-logged reports are skipped
        start, end, cap = now - timedelta(days=3), now - timedelta(minutes=20), a.max_minutes or 40
    else:
        start = datetime.fromisoformat(a.start)
        end = min(datetime.fromisoformat(a.end) + timedelta(hours=23, minutes=45), now - timedelta(minutes=20))
        cap = a.max_minutes or 330
    start = start.replace(minute=start.minute - start.minute % 15, second=0, microsecond=0)

    t0, changes, logrows, n_ok, complete = time.time(), [], [], 0, True
    for ts in timestamps(start, end, a.step):
        key = ts.strftime("%Y-%m-%d %H:%M")
        if key in done:
            continue
        if (time.time() - t0) / 60 > cap:
            complete = False
            break
        try:
            df = injury.get_reportdata(ts, return_df=True)
            rows = diff(clean(df), state, ts)
            changes += rows
            logrows.append(dict(report_et=key, mode=a.mode, result="ok", rows=len(df), note=""))
            n_ok += 1
        except URLRetrievalError:
            if a.mode == "live" and ts > now - timedelta(minutes=90):
                continue                       # may not be published yet: retry next run
            logrows.append(dict(report_et=key, mode=a.mode, result="missing", rows=0, note=""))
        except Exception as e:                 # parse problems: log and move on
            logrows.append(dict(report_et=key, mode=a.mode, result="error", rows=0,
                                note=f"{type(e).__name__}: {e}"[:200]))

    if changes:
        ch = pd.DataFrame(changes, columns=COLS)
        for season, part in ch.groupby(ch.game_date.map(lambda d: season_of(datetime.fromisoformat(d)))):
            path = f"{DATA}/status_changes_{season}.csv"
            part.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    if logrows:
        (pd.concat([log, pd.DataFrame(logrows)])
         .drop_duplicates(subset="report_et", keep="last")       # latest result per report time
         .sort_values("report_et").to_csv(LOG, index=False))
    json.dump(state, open(f"{DATA}/state_{a.mode}.json", "w"))

    res = pd.Series([r["result"] for r in logrows]).value_counts().to_dict() if logrows else {}
    print(f"{a.mode}: {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} ET | reports {res} | "
          f"{len(changes)} status changes | {'complete' if complete else 'STOPPED AT TIME CAP: run again'}")
    errs = [r for r in logrows if r["result"] == "error"]
    for r in errs[:5]:
        print("  error", r["report_et"], r["note"])
    if n_ok == 0 and errs:
        sys.exit(1)                            # every report failed to parse: make the run show red


if __name__ == "__main__":
    main()
