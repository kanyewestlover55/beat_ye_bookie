"""
Beating The Bookie - Layer 2 vs the book (first live test, 2026-27 games).

Fits the GBM on 2023-24, 2024-25 and 2025-26 before 1 Feb (the holdout stays untouched),
projects every 2026-27 player-game, then compares the model's ladder probabilities with
  (a) bet365's closing ladder prices from the feed (one-sided, vig included), and
  (b) the no-vig consensus of US books' closing Over/Under at the main line (bot capture).
Prints log loss / Brier for model and market, calibration, and what happens when they disagree.
"""
import re
import unicodedata

import numpy as np
import pandas as pd
import lightgbm as lgb

import projection_model as pm

ABBR = {"WAS": "WSH", "UTA": "UTAH"}                       # feed -> ESPN
FULL = {"Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
        "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
        "Detroit Pistons": "DET", "Golden State Warriors": "GS", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
        "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NO",
        "New York Knicks": "NY", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
        "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SA",
        "Toronto Raptors": "TOR", "Utah Jazz": "UTAH", "Washington Wizards": "WSH"}
NICK = {"alexandre sarr": "alex sarr", "carlton carrington": "bub carrington", "gregory jackson": "gg jackson"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'\-]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return NICK.get(s, s)


def slate(utc):
    return (pd.to_datetime(utc, utc=True) - pd.Timedelta(hours=10)).dt.date


def logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()


def brier(p, y):
    return ((p - y) ** 2).mean()


def fit_and_project():
    pm.SEASONS = [2023, 2024, 2025, 2026, 2027]
    p = pm.features()
    train = p[(p.season.isin([2024, 2025]) | ((p.season == 2026) & (p.game_date < pm.VAL_END)))
              & (p.career_games >= 5)]
    gbm = lgb.LGBMRegressor(objective="poisson", n_estimators=600, learning_rate=0.03, num_leaves=31,
                            min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                            verbose=-1).fit(train[pm.FEATS], train.fg3m)
    r = pm.fit_r(np.clip(gbm.predict(train[pm.FEATS]), 0.01, None), train.fg3m.values)
    live = p[p.season == 2027].copy()
    live["mu"] = np.clip(gbm.predict(live[pm.FEATS]), 0.02, None)
    live["key"] = live.player.map(norm)
    live["slate"] = pd.to_datetime(live.game_date).dt.date
    live["home_abbr"] = np.where(live.home == 1, live.team, live.opp)
    return live, r


def vs_bet365(live, r):
    cl = pd.read_csv("data/bet365/2026-10/threes_closing.csv", low_memory=False)
    cl = cl[cl.market == "ladder"].copy()
    cl["k"] = pd.to_numeric(cl.threshold)
    cl["home_abbr"] = cl.game.str.split(" @ ").str[1].str.split().str[0].replace(ABBR)
    cl["slate"] = slate(cl.kickoff_utc)
    cl["key"] = cl.player.map(norm)
    m = cl.merge(live[["slate", "home_abbr", "key", "player", "fg3m", "mu", "games_prior", "min_ewm20"]],
                 on=["slate", "home_abbr", "key"], how="inner", suffixes=("_feed", ""))
    m["hit"] = (m.fg3m >= m.k).astype(float)
    m["q"] = 1 / m.close_odds                          # book implied, vig included
    P = pm.nb_ladder(m.mu.values, r)                   # columns = 1+..4+
    m["p"] = np.nan
    for i, k in enumerate(pm.LADDER):
        m.loc[m.k == k, "p"] = P[(m.k == k).values, i]
    m = m[m.k.isin(pm.LADDER)]
    print(f"\n=== bet365 closing ladders: {m.fixture_id.nunique()} games, {m.groupby(['slate','key']).ngroups} "
          f"player-games, {len(m)} prices (1+ to 4+) ===")
    rows = []
    for k, g in m.groupby("k"):
        rows.append(dict(k=f"{k}+", n=len(g), hit_rate=g.hit.mean(), book_implied=g.q.mean(), model_p=g.p.mean(),
                         LL_book=logloss(g.q, g.hit), LL_model=logloss(g.p, g.hit),
                         Brier_book=brier(g.q, g.hit), Brier_model=brier(g.p, g.hit),
                         corr_p_q=np.corrcoef(g.p, g.q)[0, 1]))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    # disagreement: model vs book (book has vig, so compare to book minus its sample margin per k)
    m["q_adj"] = m.q - m.groupby("k").q.transform("mean") + m.groupby("k").hit.transform("mean")
    m["edge"] = m.p - m.q_adj
    m["bucket"] = pd.cut(m.edge, [-1, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 1])
    d = m.groupby("bucket", observed=True).agg(n=("hit", "size"), hit_rate=("hit", "mean"),
                                              model_p=("p", "mean"), book_adj=("q_adj", "mean"))
    print("\nWhen model and (margin-adjusted) book disagree, who is right?\n", d.round(3).to_string())
    return m


def vs_us_books(live, r):
    pr = pd.read_csv("data/closing_props.csv")
    pr = pr[pr.market == "player_threes"].copy()
    pr["slate"] = slate(pr.commence_utc)
    pr["home_abbr"] = pr.home.map(FULL)
    pr["key"] = pr.player.map(norm)
    pr["imp"] = 1 / pr.price
    w = pr.pivot_table(index=["slate", "home_abbr", "key", "book", "point"], columns="side", values="imp").dropna()
    w["p_over"] = w.Over / (w.Over + w.Under)                                   # no-vig per book
    main = w.groupby(["slate", "home_abbr", "key"]).apply(
        lambda g: g.reset_index().groupby("point").agg(p_over=("p_over", "mean"), books=("book", "size"))
        .sort_values("books", ascending=False).head(1).reset_index()).reset_index(level=3, drop=True).reset_index()
    m = main.merge(live[["slate", "home_abbr", "key", "player", "fg3m", "mu"]], on=["slate", "home_abbr", "key"])
    m["k"] = (m.point + 0.5).astype(int)
    m["hit"] = (m.fg3m >= m.k).astype(float)
    m["p"] = [pm.nb_ladder(np.array([mu]), r)[0, k - 1] if 1 <= k <= 4 else np.nan for mu, k in zip(m.mu, m.k)]
    m = m.dropna(subset=["p"])
    print(f"\n=== US-book no-vig consensus at the main line: {len(m)} player-games "
          f"({m.books.mean():.1f} books avg) ===")
    print(f"hit rate {m.hit.mean():.3f} | market p {m.p_over.mean():.3f} | model p {m.p.mean():.3f}")
    print(f"log loss  market {logloss(m.p_over, m.hit):.4f}  model {logloss(m.p, m.hit):.4f}  "
          f"50/50 {logloss(np.full(len(m), .5), m.hit):.4f}")
    print(f"Brier     market {brier(m.p_over, m.hit):.4f}  model {brier(m.p, m.hit):.4f}")
    print(f"corr(model, market) {np.corrcoef(m.p, m.p_over)[0,1]:.3f}")
    blend = 0.5 * m.p + 0.5 * m.p_over
    print(f"50/50 blend log loss {logloss(blend, m.hit):.4f}")
    m["edge"] = m.p - m.p_over
    m["bucket"] = pd.cut(m.edge, [-1, -0.1, -0.05, 0.05, 0.1, 1])
    print(m.groupby("bucket", observed=True).agg(n=("hit", "size"), hit_rate=("hit", "mean"), model_p=("p", "mean"),
                                                 market_p=("p_over", "mean")).round(3).to_string())
    return m


if __name__ == "__main__":
    live, r = fit_and_project()
    print(f"2026-27 player-games projected: {len(live)} over {live.game_id.nunique()} games; NB size r = {r:.1f}")
    a = vs_bet365(live, r)
    b = vs_us_books(live, r)
    a.to_csv("l2_vs_bet365.csv", index=False)
    b.to_csv("l2_vs_usbooks.csv", index=False)
