"""
Beating The Bookie - one-off probe of the MoviOdds bet365 feed (no key needed).
 
Saves to data/probe/:
  board_nba.json      current bet365 NBA board (all markets)
  summary.txt         NBA matches, market names found, which look like 3PM markets
  history_<id>.json   price history for recently finished NBA games (3PM + game markets only)
  closing_<id>.json   bet365 closing prices for the same games
"""
import datetime as dt
import json
import pathlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request
 
BASE = "https://moviodds.com"
OUT = pathlib.Path("data/probe")
OUT.mkdir(parents=True, exist_ok=True)
THREES = re.compile(r"three|3[- ]?point|threes|made 3|3pt", re.I)
GAME_GROUPS = {"1453", "112"}
log = []
 
 
def say(msg):
    print(msg)
    log.append(str(msg))
 
 
def get(path, **params):
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "beat-ye-bookie-probe/1.0",
                                                       "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                time.sleep(int(e.headers.get("Retry-After", 10)))
                continue
            say(f"HTTP {e.code} on {path} {params}")
            return None
        except Exception as e:
            say(f"Error on {path}: {e}")
            time.sleep(5)
    return None
 
 
def is_nba(m):
    return (m.get("league") or "").strip().upper() == "NBA"
 
 
# 1. Current NBA board
board = get("/odds", sport="basketball") or {}
nba = [m for m in board.get("matches", []) if is_nba(m)]
(OUT / "board_nba.json").write_text(json.dumps(nba, indent=1))
say(f"Current bet365 NBA matches: {len(nba)}")
groups = {}
for m in nba:
    say(f"  {m.get('away')} @ {m.get('home')}  kickoff {m.get('kickoff_utc')}  "
        f"markets={m.get('market_count')}  quotes={len(m.get('quotes', []))}")
    for q in m.get("quotes", []):
        key = (q.get("group_id"), q.get("group_name"))
        groups[key] = groups.get(key, 0) + 1
say("\nMarket groups on the board (id | name | price count):")
for (gid, name), n in sorted(groups.items(), key=lambda x: -x[1]):
    flag = "   <-- looks like 3PM" if THREES.search(name or "") else ""
    say(f"  {gid} | {name} | {n}{flag}")
 
# 2. Recently finished NBA games, with price history and closing lines
today = dt.datetime.now(dt.timezone.utc).date()
finished = []
for back in (1, 2):
    day = (today - dt.timedelta(days=back)).isoformat()
    res = get("/results", sport="basketball", date=day) or {}
    finished += [r for r in res.get("results", []) if is_nba(r)]
say(f"\nFinished NBA games found (last 2 UTC days): {len(finished)}")
 
for r in finished[:14]:
    fi = r["fixture_id"]
    say(f"  {r.get('away')} @ {r.get('home')}  final {r.get('final_score')}  fi={fi}")
    hist = get("/history", fi=fi, limit=50000) or {}
    pts = hist.get("points", [])
    names = sorted({p.get("group_name") or "" for p in pts})
    keep = [p for p in pts if THREES.search(p.get("group_name") or "") or p.get("group_id") in GAME_GROUPS]
    (OUT / f"history_{fi}.json").write_text(json.dumps(
        dict(fixture=r, total_points=len(pts), truncated=hist.get("truncated"),
             group_names=names, points=keep), indent=1))
    say(f"     history: {len(pts)} price points, {len(keep)} kept (3PM + game lines); "
        f"markets: {', '.join(n for n in names if THREES.search(n)) or 'no 3PM-looking market'}")
    clo = get("/closing", fi=fi)
    if clo:
        (OUT / f"closing_{fi}.json").write_text(json.dumps(clo, indent=1))
    time.sleep(1)
 
(OUT / "summary.txt").write_text("\n".join(log) + "\n")
 
