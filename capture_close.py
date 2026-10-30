"""
Beating The Bookie - closing odds capture (runs on GitHub Actions every 15 min).
 
For every NBA game tipping within the next WINDOW_MIN minutes that hasn't been
captured yet, it saves:
  - data/closing_props.csv : 3PM props (two-sided Over/Under, so no-vig is possible)
  - data/closing_lines.csv : spreads and totals
Budget (free plan, 500 credits/month): events list = free, props = 1 credit per game
(~225/month), spreads+totals = 2 credits per call, batched for all games tipping in the
next 4 hours (~2 calls a night, ~120/month). Expect ~350 credits/month.
Credits remaining are logged to data/credits_log.csv after every paid call.
"""
import csv
import datetime as dt
import json
import os
import pathlib
import urllib.parse
import urllib.request
 
API = "https://api.the-odds-api.com/v4/sports/basketball_nba"
KEY = os.environ["ODDS_API_KEY"]
REGION = "us"
PROP_MARKETS = os.environ.get("PROP_MARKETS", "player_threes")  # add ,player_threes_alternate if budget allows
WINDOW_MIN = 45          # capture games tipping within this many minutes (widened from 30 on 29 Oct: GitHub skips runs)
MIN_CREDITS_LEFT = 15    # stop spending if the monthly quota is nearly used
 
DATA = pathlib.Path("data")
DATA.mkdir(exist_ok=True)
DONE = DATA / "captured_events.txt"
LINES_DONE = DATA / "lines_captured.txt"
LINES_AHEAD_MIN = 240    # batch game lines for all games tipping in the next 4 hours
 
LINE_COLS = ["captured_utc", "minutes_to_tip", "event_id", "commence_utc", "away", "home",
             "book", "market", "selection", "point", "price"]
PROP_COLS = ["captured_utc", "minutes_to_tip", "event_id", "commence_utc", "away", "home",
             "book", "market", "player", "side", "point", "price"]
 
 
def get(path, **params):
    params["apiKey"] = KEY
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r), r.headers.get("x-requests-remaining")
 
 
def log_credits(remaining, note):
    append(DATA / "credits_log.csv",
           [dict(captured_utc=stamp, credits_remaining=remaining, note=note)],
           ["captured_utc", "credits_remaining", "note"])
 
 
def append(path, rows, cols):
    if not rows:
        return
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerows(rows)
 
 
now = dt.datetime.now(dt.timezone.utc)
stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
done = set(DONE.read_text().split()) if DONE.exists() else set()
lines_done = set(LINES_DONE.read_text().split()) if LINES_DONE.exists() else set()
 
events, _ = get("events")  # free call
print(f"{stamp} | {len(events)} upcoming NBA events. Next tips:")
due, line_ids, mins_by_id = [], [], {}
for e in sorted(events, key=lambda x: x["commence_time"]):
    tip = dt.datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
    mins = (tip - now).total_seconds() / 60
    if mins > 0 and len(due) < 50:
        print(f"  {e['away_team']} @ {e['home_team']}  tip {e['commence_time']}  ({mins:.0f} min)")
    mins_by_id[e["id"]] = round(mins, 1)
    if 0 < mins <= WINDOW_MIN and e["id"] not in done:
        due.append((e, mins))
    if 0 < mins <= LINES_AHEAD_MIN and e["id"] not in lines_done:
        line_ids.append(e["id"])
 
if not due:
    print("No games due for capture this run.")
    raise SystemExit(0)
 
remaining = None
line_rows = []
lines = []
if line_ids:
    lines, remaining = get("odds", regions=REGION, markets="spreads,totals",
                           oddsFormat="decimal", eventIds=",".join(line_ids))
    lines_done.update(line_ids)
    LINES_DONE.write_text("\n".join(sorted(lines_done)) + "\n")
    log_credits(remaining, f"lines for {len(line_ids)} games")
for g in lines:
    for b in g.get("bookmakers", []):
        for m in b.get("markets", []):
            for o in m.get("outcomes", []):
                line_rows.append(dict(captured_utc=stamp, minutes_to_tip=mins_by_id.get(g["id"]),
                                      event_id=g["id"], commence_utc=g["commence_time"],
                                      away=g["away_team"], home=g["home_team"], book=b["key"],
                                      market=m["key"], selection=o["name"],
                                      point=o.get("point"), price=o["price"]))
append(DATA / "closing_lines.csv", line_rows, LINE_COLS)
 
prop_rows = []
for e, mins in due:
    if remaining is not None and int(float(remaining)) < MIN_CREDITS_LEFT:
        print(f"Only {remaining} credits left - skipping props to protect the quota.")
        break
    try:
        g, remaining = get(f"events/{e['id']}/odds", regions=REGION,
                           markets=PROP_MARKETS, oddsFormat="decimal")
    except Exception as ex:
        print(f"Props failed for {e['away_team']} @ {e['home_team']}: {ex}")
        continue
    for b in g.get("bookmakers", []):
        for m in b.get("markets", []):
            for o in m.get("outcomes", []):
                prop_rows.append(dict(captured_utc=stamp, minutes_to_tip=round(mins, 1),
                                      event_id=g["id"], commence_utc=g["commence_time"],
                                      away=g["away_team"], home=g["home_team"], book=b["key"],
                                      market=m["key"], player=o.get("description"),
                                      side=o["name"], point=o.get("point"), price=o["price"]))
    done.add(e["id"])
    log_credits(remaining, f"props {e['away_team']} @ {e['home_team']}")
 
append(DATA / "closing_props.csv", prop_rows, PROP_COLS)
DONE.write_text("\n".join(sorted(done)) + "\n")
print(f"Captured {len(due)} game(s): {len(line_rows)} line rows, {len(prop_rows)} prop rows. "
      f"Credits remaining this month: {remaining}")
 
