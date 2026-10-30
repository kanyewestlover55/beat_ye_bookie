"""
Beating The Bookie - bet365 collector (MoviOdds, free, no key). Runs on GitHub Actions.
 
Every run:
  1. Records tonight's NBA fixtures from the bet365 board (ids, teams, tip times).
  2. For every NBA game that finished (tipped 4+ hours ago) and isn't collected yet, saves:
     - the full pre-match price timeline for bet365's 3PM ladder (1+, 2+ ...) and 3PM Over/Under
     - the same for the main game lines (spread, total, moneyline)
     - bet365's closing price for each of those selections
Files (one folder per month, so no file grows too large):
  data/bet365/<YYYY-MM>/threes_history.csv, threes_closing.csv, lines_history.csv
  data/bet365/fixtures.csv, collected.txt, run_log.txt
MoviOdds is an unofficial scraper and may break; screenshots remain the fallback.
"""
import csv
import datetime as dt
import json
import pathlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request
 
BASE = "https://moviodds.com"
ROOT = pathlib.Path("data/bet365")
ROOT.mkdir(parents=True, exist_ok=True)
FIXTURES = ROOT / "fixtures.csv"
COLLECTED = ROOT / "collected.txt"
GROUPS = {"181449": "ladder", "181424": "ou", "1453": "lines"}   # pre-match group ids
MIN_HOURS_AFTER_TIP = 4
NOW = dt.datetime.now(dt.timezone.utc)
STAMP = NOW.strftime("%Y-%m-%dT%H:%M:%SZ")
log = []
 
HIST_COLS = ["fixture_id", "game", "kickoff_utc", "market", "player", "threshold", "side",
             "ts", "odds_dec", "suspended", "group_id", "sel_id", "raw_label", "raw_side",
             "raw_line", "raw_handicap"]
CLOSE_COLS = ["fixture_id", "game", "kickoff_utc", "market", "player", "threshold", "side",
              "close_ts", "close_odds", "close_basis", "group_id", "sel_id", "raw_label",
              "raw_side", "raw_line", "raw_handicap"]
FIX_COLS = ["fixture_id", "away", "home", "kickoff_utc", "first_seen_utc"]
 
 
def say(msg):
    print(msg)
    log.append(str(msg))
 
 
def get(path, **params):
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    for _ in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "beat-ye-bookie/1.0",
                                                       "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                time.sleep(int(e.headers.get("Retry-After", 10) or 10))
                continue
            say(f"HTTP {e.code} on {path} {params}")
            return None
        except Exception as e:
            say(f"Error on {path} {params}: {e}")
            time.sleep(5)
    return None
 
 
def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
 
 
def normalise(p, market):
    """Field use differs by market and endpoint, so find player/threshold/side by content."""
    if market == "lines":   # bet type in `line` (Line / Total / To Win), team or Over/Under in `side`
        n = [num(f) for f in (p.get("label"), p.get("handicap")) if num(f) is not None]
        if not n and p.get("label"):
            n = [num(x) for x in re.findall(r"[-+]?\d+\.?\d*", p["label"])]
        return p.get("line") or "", (n[0] if n else ""), p.get("side") or ""
    fields = [p.get("label"), p.get("side"), p.get("line"), p.get("handicap")]
    side = next((f for f in fields if f in ("Over", "Under", "O", "U")), "")
    side = {"O": "Over", "U": "Under"}.get(side, side)
    player = next((f for f in fields if f and num(f) is None and f not in ("Over", "Under", "O", "U")), "")
    nums = [num(f) for f in fields if num(f) is not None]
    threshold = ""
    if nums:
        half = [x for x in nums if abs(x % 1 - 0.5) < 1e-9]
        if market == "ladder":   # "3+" may arrive as 3 or as line 2.5
            threshold = int(half[0] + 0.5) if half else int(max(nums))
        else:                    # O/U and game lines keep the line, e.g. 2.5
            threshold = half[0] if half else nums[0]
    return player, threshold, side
 
 
def raw(p):
    return dict(raw_label=p.get("label"), raw_side=p.get("side"),
                raw_line=p.get("line"), raw_handicap=p.get("handicap"))
 
 
def append(path, rows, cols):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)
 
 
def is_nba(m):
    return (m.get("league") or "").strip().upper() == "NBA"
 
 
# 1. Record upcoming NBA fixtures from the board (game lines only, keeps the response small)
known = {}
if FIXTURES.exists():
    with FIXTURES.open() as f:
        known = {r["fixture_id"]: r for r in csv.DictReader(f)}
board = get("/odds", sport="basketball", groups="1453") or {}
new_fix = []
for m in board.get("matches", []):
    if is_nba(m) and m["id"] not in known:
        row = dict(fixture_id=m["id"], away=m.get("away"), home=m.get("home"),
                   kickoff_utc=m.get("kickoff_utc"), first_seen_utc=STAMP)
        known[m["id"]] = row
        new_fix.append(row)
append(FIXTURES, new_fix, FIX_COLS)
say(f"{STAMP} | board: {len(board.get('matches', []))} basketball matches, {len(new_fix)} new NBA fixtures recorded")
 
# Also pick up finished NBA games from /results (covers anything the board run missed)
for back in range(0, 3):
    day = (NOW.date() - dt.timedelta(days=back)).isoformat()
    for r in (get("/results", sport="basketball", date=day) or {}).get("results", []):
        if is_nba(r) and r["fixture_id"] not in known:
            row = dict(fixture_id=r["fixture_id"], away=r.get("away"), home=r.get("home"),
                       kickoff_utc="", first_seen_utc=STAMP)
            known[r["fixture_id"]] = row
            append(FIXTURES, [row], FIX_COLS)
 
# 2. Collect finished games
done = set(COLLECTED.read_text().split()) if COLLECTED.exists() else set()
todo = []
for fid, f in known.items():
    if fid in done:
        continue
    ko = f.get("kickoff_utc")
    if ko:
        tip = dt.datetime.fromisoformat(ko.replace("Z", "+00:00"))
        if (NOW - tip).total_seconds() < MIN_HOURS_AFTER_TIP * 3600:
            continue
    todo.append(f)
say(f"Games to collect this run: {len(todo)}")
 
for f in todo:
    fid = f["fixture_id"]
    game = f"{f.get('away')} @ {f.get('home')}"
    clo = get("/closing", fi=fid)
    if not clo:
        say(f"  {game}: no closing data yet, will retry next run")
        continue
    ko = clo.get("kickoff_utc") or f.get("kickoff_utc") or ""
    month = (ko or STAMP)[:7]
    folder = ROOT / month
 
    close_rows = []
    for s in clo.get("selections", []):
        market = GROUPS.get(str(s.get("group_id")))
        pc = s.get("pre_close")
        if not market or not pc:
            continue
        player, thr, side = normalise(s, market)
        close_rows.append(dict(fixture_id=fid, game=game, kickoff_utc=ko, market=market,
                               player=player, threshold=thr, side=side, close_ts=pc.get("ts"),
                               close_odds=pc.get("odds_dec"), close_basis=clo.get("pre_close_basis"),
                               group_id=s.get("group_id"), sel_id=s.get("sel_id"), **raw(s)))
 
    hist_rows = {"threes": [], "lines": []}
    truncated = []
    for gid, market in GROUPS.items():
        h = get("/history", fi=fid, group=gid, limit=50000, **({"to": ko} if ko else {})) or {}
        if h.get("truncated"):
            truncated.append(market)
        for p in h.get("points", []):
            player, thr, side = normalise(p, market)
            row = dict(fixture_id=fid, game=game, kickoff_utc=ko, market=market, player=player,
                       threshold=thr, side=side, ts=p.get("ts"), odds_dec=p.get("odds_dec"),
                       suspended=p.get("suspended"), group_id=p.get("group_id"),
                       sel_id=p.get("sel_id"), **raw(p))
            hist_rows["lines" if market == "lines" else "threes"].append(row)
        time.sleep(1)
 
    append(folder / "threes_history.csv", hist_rows["threes"], HIST_COLS)
    append(folder / "lines_history.csv", hist_rows["lines"], HIST_COLS)
    append(folder / "threes_closing.csv", [r for r in close_rows if r["market"] != "lines"], CLOSE_COLS)
    append(folder / "lines_closing.csv", [r for r in close_rows if r["market"] == "lines"], CLOSE_COLS)
    players = len({r["player"] for r in close_rows if r["market"] == "ladder"})
    n_ou = sum(r["market"] == "ou" for r in hist_rows["threes"])
    ou_players = len({r["player"] for r in hist_rows["threes"] if r["market"] == "ou"})
    say(f"  {game}: {len(hist_rows['threes'])} 3PM price points ({n_ou} O/U, {ou_players} O/U players), "
        f"{len(close_rows)} closes, {players} players" + (f", TRUNCATED: {truncated}" if truncated else ""))
    if close_rows or hist_rows["threes"]:
        done.add(fid)
    time.sleep(1)
 
COLLECTED.write_text("\n".join(sorted(done)) + "\n")
with (ROOT / "run_log.txt").open("a") as fh:
    fh.write("\n".join(log) + "\n\n")
 
