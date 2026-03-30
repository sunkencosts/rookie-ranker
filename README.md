ML project to rank NFL rookies for fantasy drafts using college stats and draft position.

# Contributors
- SunkenCosts
- nloughl

## Whiteboard
https://excalidraw.com/#room=73154c6bed19c2a1a4f3,kyYFC_0O40MEpfrJR0dAaA

---

## How it works
Predicts a rookie's fantasy points in their first NFL season based on their final college season stats and draft pick number.

**Training data:** 5 years (2020–2024 college → 2021–2025 NFL rookie seasons), ~1086 players
**Models:** Two variants — one with draft pick, one without (for pre-draft use)
**Current accuracy:** CV R² = 0.41 (with pick), 0.24 (no pick)

## Usage
```bash
# Fetch data + train both models
python3 main.py

# Force re-fetch all data from APIs
python3 main.py --force

# Rank current rookie class (pre-draft, no pick number)
python3 main.py --predict
```

## Done
- [x] College stats pipeline (collegefootballdata.com API, 2020–2025)
- [x] NFL draft + rookie fantasy points pipeline (nflreadpy, 2021–2025)
- [x] Fuzzy name matching to join college and NFL data
- [x] Two models: pre-draft (no pick) and post-draft (with pick)
- [x] 5-fold cross-validation for reliable accuracy measurement
- [x] Pre-draft prospect filtering by position stat thresholds
- [x] Caching at every stage — API calls only happen once

## TODO
- [ ] Wire up post-draft `--predict` using pick numbers (draft is late April 2026)
- [ ] Dominator rating — what % of team receiving yards/TDs a player accounted for
- [ ] Multi-year training data beyond 5 years

## Dynasty model (future)
Goal: predict who will be the best asset in 3–5 years — the next Josh Allen, Nico Collins, JSN.

**Target variable:** 3-year cumulative fantasy points instead of rookie year only.

**Key features to add:**
- Breakout age — age at which a player first dominated their college team's production (earlier = higher ceiling)
- Dominator rating — % of team receiving yards + TDs (requires CFD team stats API)
- Multi-year college trajectory — not just final season, but sophomore/junior trends

**Tradeoff:** using a 3-year target drops the 2024 and 2025 draft classes from training data (not enough NFL seasons yet), reducing ~1086 rows to ~650. Can be offset by extending training back to 2015 draft class.

**Approach:** hard 3-year cutoff on training data. Cleaner target variable is worth the smaller dataset.
