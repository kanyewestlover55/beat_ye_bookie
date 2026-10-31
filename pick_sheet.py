"""
Beating The Bookie - evening pick sheet.

Reads the paper bot's output and writes <folder>/tonight.md: tonight's picks grouped by game, with Swiss tip
times, the odds the bot saw, and the minimum odds still worth taking (the model's break-even price, 1 / model
probability, rounded up). Read-only on the bot's files: it changes nothing in the frozen test.

  python3 pick_sheet.py [data/paper]        (does nothing if the folder or its files don't exist)
"""
import glob
import math
import pathlib
import sys

import pandas as pd

SWISS = "Europe/Zurich"


def swiss(ts):
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    return t.tz_convert(SWISS)


def main(folder):
    d = pathlib.Path(folder)
    cands = sorted(glob.glob(str(d / "candidates_*.csv")))
    if not cands:
        return
    c = pd.concat([pd.read_csv(f) for f in cands])
    last = c.decision_utc.max()
    c = c[c.decision_utc == last]
    picks = pd.read_csv(d / "picks.csv") if (d / "picks.csv").exists() else pd.DataFrame()
    p = picks[picks.decision_utc == last].copy() if len(picks) else picks
    dec = swiss(last)
    lines = [f"# Paper picks, {dec:%a %d %b}",
             "",
             f"Decided {dec:%H:%M} Swiss · {c.fixture_id.nunique()} games · {len(c)} prices checked · "
             f"**{len(p)} bets** · injuries: {c.injury_source.iloc[0]}",
             ""]
    if p.empty:
        lines += ["No bets tonight: no price met the rule.", ""]
    else:
        p["tip"] = p.kickoff_utc.map(swiss)
        p["min_odds"] = (1 / p.p_model).map(lambda x: math.ceil(x * 100) / 100)
        for (game, tip), g in p.sort_values(["tip", "game", "player", "k"]).groupby(["game", "tip"], sort=False):
            age = g.quote_age_min.median()
            lines += [f"## {game}", f"Tip {tip:%a %H:%M} Swiss · prices about {age:.0f} min old when read", "",
                      "| Player | Bet | Odds seen | Min odds | Model | EV |", "| --- | --- | --- | --- | --- | --- |"]
            for r in g.itertuples():
                lines.append(f"| {r.player} | {int(r.k)}+ threes | {r.odds:.2f} | {r.min_odds:.2f} | "
                             f"{100 * r.p_model:.0f}% | {100 * r.ev:+.0f}% |")
            outs = sorted({o.strip() for x in g.known_out.dropna() for o in str(x).split(",") if o.strip()})
            if outs:
                lines += ["", f"Teammates out (known at decision time): {', '.join(outs)}"]
            lines.append("")
    lines += ["---",
              "**Odds seen:** bet365's price on the feed when the bot ran (can lag the app by up to an hour).  ",
              "**Min odds:** the lowest price the model still rates as a bet. The app cuts decimals (1.689 shows as 1.68), "
              "so an app price 0.01 below the minimum may still be fine.  ",
              "**Spot check (paper phase):** compare a few *Odds seen* with the app and send a screenshot in the weekly chat.  ",
              "**Stage 4 (real bets, only after the paper test passes):** bet a pick only if the app shows at least the min odds; "
              "1 unit = 0.5% of the bankroll; log the odds you actually got.",
              ""]
    (d / "tonight.md").write_text("\n".join(lines))
    print(f"wrote {d / 'tonight.md'}: {len(p)} bets from the {dec:%Y-%m-%d %H:%M} Swiss run")


if __name__ == "__main__":
    for folder in (sys.argv[1:] or ["data/paper"]):
        if pathlib.Path(folder).exists():
            main(folder)
