# xGRAPM — Expected Goal Regularized Adjusted Plus-Minus

Interactive football analytics dashboard and two-stage player valuation pipeline calibrated on **1,826 matches from the 2015/16 Big 5 European Leagues**.

## Methodology Overview
- **Stage 1 (Micro — Markovian State Value):** Evaluates discrete player turns on a 120x80m pitch using Euclidean distance and visible shooting angle to the goalposts, compounding forward 10-action shot probabilities and separating **In-Possession Creation (Micro Off)** from **Out-of-Possession Ball-Winning (Micro Def)**.
- **Stage 2 (Macro — Ridge Regression):** Solves for off-ball positioning and team tactical structure across lineup stints while controlling for home advantage, red-card manpower differentials, and score-state effects.
- **Dashboard Modules (`index.html` + `main.py`):**
  1. **Scouting & Plays:** Filterable player leaderboard with Position-Relative Role Percentiles, Per-Touch Efficiency (`npxG / 100 actions`), and animated progressive play sequences.
  2. **Match Context Lab:** Head-to-head Starting XI lineup simulator with manpower and venue adjustments.
  3. **2016 Ballon d'Or vs. xGRAPM:** Empirical comparison against the 30-man *France Football* ballot.
  4. **Threat Heatmap:** 12x8 spatial-angular danger grid and Pitch Depth x Angle Corridor Matrix across 8 tactical game states.

## Run Locally
- Install dependencies: `pip install -r requirements.txt`
- Start server: `uvicorn main:app --reload`
- Open `http://127.0.0.1:8000` in your browser.
