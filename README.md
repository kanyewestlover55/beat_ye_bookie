# Beating The Bookie

Learning project: a rules-based system for NBA player props (3-pointers made first, tennis later),
measured by closing line value (CLV). Paper trading only until the testing roadmap is passed.
The full rulebook, findings and decisions live in the Claude Doc "Beating The Bookie";
this repo holds the code and the data.

## Start of a new chat (for Claude)

1. Read this README.
2. Download `odds_log.csv` and everything in `data/` from this repo (public, raw.githubusercontent.com).
3. Rebuild the datasets: `pip install pyarrow lightgbm`, then run `build_dataset.py` and `build_game_lines.py`
   (both download their own sources), then `projection_model.py` (Layer 2, needs the other two first).
4. Open the Claude Doc for the rulebook, status table and findings log.

## What's here

| File | What it is | Updated by |
| --- | --- | --- |
| `odds_log.csv` | bet365 (user's Swiss book) 3PM ladders 1+ to 5+ and game lines, transcribed from morning/evening screenshots | Uploaded after each logging session |
| `data/closing_props.csv` | Market closing 3PM Over/Under (US books) within 30 min of tip | GitHub Actions bot, automatically |
| `data/closing_lines.csv` | Spreads and totals for the same games | Bot |
| `data/credits_log.csv` | Odds API credits remaining (free plan: 500/month) | Bot |
| `capture_close.py` | The bot's script | Manual |
| `.github/workflows/capture.yml` | Bot schedule: every 15 min, 16:00-05:59 UTC | Manual |
| `build_dataset.py` | Player-game dataset with pre-game-only features (sportsdataverse / ESPN box scores) | Manual |
| `build_game_lines.py` | Historical totals and spreads matched to ESPN games | Manual |
| `projection_model.py` | Layer 2: 3PM projection (cross-season features, structured + GBM), validation report | Manual |
| `layer2_vs_book.py` | Layer 2 vs the book: scores the model's ladder probabilities against bet365 closes and US no-vig closes on this season's games | Manual |
| `data/injuries/status_changes_<season>.csv` | Official NBA injury report as a change log: one row each time a player is added, changes status (Out, Questionable...) or is removed, with the report time | Injury bot, every 3 hours |
| `data/injuries/reports_log.csv` | Every injury report time the bot tried: ok / missing / error | Injury bot |
| `news_test.py` | News test: injury report changes vs bet365 ladder moves and outcomes (Layer 2, 31 Oct) | Manual |
| `collect_injuries.py` | Injury bot script (reads the NBA's PDF archive via the `nbainjuries` package; catches up on anything missed) | Manual |
| `.github/workflows/injuries.yml` | Injury bot schedule (every 3 hours) and manual backfill by date range | Manual |

## Conventions

- One slate = one US game night. The book dates late tips by Swiss time (a Friday US game shows as Saturday 01:10).
- Capture times are Swiss local time in `odds_log.csv`, UTC in `data/` (injury files carry both US Eastern and UTC report times).
- bet365 "3+" is the same bet as US books' "Over 2.5".
