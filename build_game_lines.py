"""
Beating The Bookie - historical game lines (totals, spreads, moneylines).

Source: public odds database in github.com/kyleskom/NBA-Machine-Learning-Sports-Betting
Matched to ESPN games (sportsdataverse) by date + home team.
Coverage: 2023-24 regular season, 2024-25 incl. playoffs, 2025-26 up to 7 Jan 2026.
Output: game_lines.parquet with total, home_fav_by (positive = home favoured),
        implied team points, moneylines.
"""
import sqlite3
import urllib.request

import pandas as pd

ODDS_DB = ("https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/raw/master/"
           "Data/OddsData.sqlite")
BOX = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
       "espn_nba_player_boxscores/player_box_{s}.parquet")
TABLES = ["2023-24", "2024-25", "odds_2025-26"]
SEASONS = [2024, 2025, 2026]


def build():
    urllib.request.urlretrieve(ODDS_DB, "OddsData.sqlite")
    con = sqlite3.connect("OddsData.sqlite")
    odds = pd.concat([pd.read_sql(f'select Date, Home, Away, OU, Spread, ML_Home, ML_Away from "{t}"', con)
                      for t in TABLES])
    odds["Date"] = pd.to_datetime(odds["Date"]).dt.date

    box = pd.concat([pd.read_parquet(BOX.format(s=s),
                                     columns=["game_id", "game_date", "team_display_name", "home_away", "season"])
                     for s in SEASONS]).drop_duplicates(["game_id", "home_away"])
    home = box[box.home_away == "home"].rename(columns={"team_display_name": "Home"})
    home["Date"] = pd.to_datetime(home["game_date"]).dt.date

    g = home.merge(odds, on=["Date", "Home"], how="left")
    g = g[["game_id", "season", "Date", "Home", "Away", "OU", "Spread", "ML_Home", "ML_Away"]]
    g = g.rename(columns={"OU": "total", "Spread": "home_fav_by"})
    g["home_implied_pts"] = (g.total + g.home_fav_by) / 2
    g["away_implied_pts"] = (g.total - g.home_fav_by) / 2
    return g


if __name__ == "__main__":
    lines = build()
    lines.to_parquet("game_lines.parquet", index=False)
    print(lines.groupby("season").total.agg(["count", "mean"]).round(1))
