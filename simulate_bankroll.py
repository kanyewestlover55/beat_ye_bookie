"""
Beating The Bookie - stage 2 of the testing roadmap: simulate before real money.

Uses the bet profile of the frozen strategy (evening decision, 1+ and 2+, model probability >= implied):
the 283 backtest bets over 9 nights, about 31 a night, median odds 1.74. Whole nights are resampled, and a
player's 1+ and 2+ bets on the same night are linked (if 2+ hits, 1+ hits). For each assumed TRUE edge
(expected return per bet), every bet's real win chance is set to (1 + edge) / odds.

Answers: how much the bankroll swings, how often a season still loses even with an edge, how deep the
drawdowns get, how often a bankroll is halved, and how many bets it takes before results mean anything.
Simplification: bets in the same game are otherwise independent, so swings are, if anything, understated.
"""
import numpy as np
import pandas as pd

SEED, SEASONS = 7, 4000
NIGHTS = 150                      # roughly November to mid-April
EDGES = [-0.05, -0.02, 0.0, 0.02, 0.05]
UNITS = {"0.5%": 200, "1%": 100, "2%": 50}     # unit as % of starting bankroll -> bankroll in units


def nights_from_backtest():
    f = pd.read_csv("absence_vs_bet365.csv")
    e = f[f.decision.str.startswith("19:30") & (f.p_v3 >= f.qm)]
    out = []
    for _, g in e.groupby("slate"):
        grp = g.groupby(["fixture_id", "key"]).ngroup().values
        out.append((g.mo.values.astype(float), grp))
    return out


def simulate(nights, edge, rng):
    """Return per-night P/L (units) and bet counts, shape (SEASONS, NIGHTS)."""
    pl = np.zeros((SEASONS, NIGHTS))
    nb = np.zeros((SEASONS, NIGHTS), dtype=int)
    pick = rng.integers(0, len(nights), size=(SEASONS, NIGHTS))
    for i, (odds, grp) in enumerate(nights):
        mask = pick == i
        n = mask.sum()
        if n == 0:
            continue
        p = np.clip((1 + edge) / odds, 0, 0.99)
        u = rng.random((n, grp.max() + 1))[:, grp]          # one draw per player: rungs move together
        ret = np.where(u < p, odds - 1, -1.0).sum(axis=1)
        pl[mask], nb[mask] = ret, len(odds)
    return pl, nb


def at_bets(cum_pl, cum_n, n):
    idx = np.argmax(cum_n >= n, axis=1)
    return cum_pl[np.arange(len(idx)), idx]


def main():
    rng = np.random.default_rng(SEED)
    nights = nights_from_backtest()
    allodds = np.concatenate([o for o, _ in nights])
    print(f"Bet profile: {len(allodds)} bets over {len(nights)} nights ({len(allodds) / len(nights):.0f} a night), "
          f"median odds {np.median(allodds):.2f}; season = {NIGHTS} nights, about {NIGHTS * len(allodds) // len(nights):,} bets\n")
    rows, dd_rows = [], []
    for edge in EDGES:
        pl, nb = simulate(nights, edge, rng)
        cum, cn = pl.cumsum(axis=1), nb.cumsum(axis=1)
        r = {"true edge": f"{edge:+.0%}"}
        for n in (300, 1000):
            x = at_bets(cum, cn, n)
            r[f"after {n}: median u"] = np.median(x)
            r[f"after {n}: 5%-95% u"] = f"{np.percentile(x, 5):+.0f} to {np.percentile(x, 95):+.0f}"
            r[f"after {n}: P(down)"] = (x < 0).mean()
        season = cum[:, -1]
        r["season median u"] = np.median(season)
        r["season P(down)"] = (season < 0).mean()
        rows.append(r)
        peak = np.maximum.accumulate(np.concatenate([np.zeros((SEASONS, 1)), cum], axis=1), axis=1)[:, 1:]
        mdd = (peak - cum).max(axis=1)
        d = {"true edge": f"{edge:+.0%}", "max drawdown median u": np.median(mdd),
             "max drawdown 95th pct u": np.percentile(mdd, 95),
             "worst night 1% u": np.percentile(pl, 1)}
        for label, bank in UNITS.items():
            d[f"halved @ {label}"] = (cum.min(axis=1) <= -bank / 2).mean()
            d[f"bust @ {label}"] = (cum.min(axis=1) <= -bank).mean()
        dd_rows.append(d)
    pd.set_option("display.width", 250)
    print("Profit in units (1 unit = 1 stake)\n", pd.DataFrame(rows).round(2).to_string(index=False))
    print("\nDrawdowns and bankroll risk over a full season\n", pd.DataFrame(dd_rows).round(3).to_string(index=False))
    sd = np.sqrt(np.mean([((1 / o) * (o - 1) ** 2 + (1 - 1 / o)) - 0 for o in allodds]))   # per-bet sd at zero edge
    for target in (0.02, 0.03, 0.05):
        print(f"Bets needed for ROI alone to show a {target:.0%} edge (2 standard errors): about {int((2 * sd / target) ** 2):,}")


if __name__ == "__main__":
    main()
