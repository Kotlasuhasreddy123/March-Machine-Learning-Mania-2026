# March Machine Learning Mania 2026

My solution for the [Kaggle March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026) competition.

**Goal:** Predict the win probability for every possible NCAA Men's and Women's tournament matchup. Scored by Brier score (lower is better).

---

## Model Overview

A logistic regression ensemble trained on historical NCAA tournament games (2010–2025), using 7 features per matchup:

| Feature | Description |
|---|---|
| Elo difference | Margin-of-victory weighted Elo ratings built from all regular season + tourney games |
| Seed difference | Tournament seed number gap (lower seed = better team) |
| Offensive rating diff | Points scored per 100 possessions (last 3 seasons) |
| Defensive rating diff | Points allowed per 100 possessions (last 3 seasons) |
| eFG% difference | Effective field goal percentage gap |
| Turnover rate diff | Turnover rate gap (lower is better) |
| Massey ordinal diff | Median rank across 60+ third-party rating systems (men's only) |

Final probabilities are passed through isotonic regression calibration and clipped to [0.025, 0.975].

Separate models are trained for Men's (TeamID < 3000) and Women's (TeamID >= 3000).

---

## Setup

1. Download the competition data from Kaggle and place all CSV files in this directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the model:
   ```bash
   python predict.py
   ```
4. Upload the generated `submission.csv` to Kaggle.

> **Note:** The competition data files are not included in this repo. You must download them from the [competition data page](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data) after accepting the competition rules.

---

## Results

- 132,133 total predictions (66,430 men's + 65,703 women's)
- Prediction range: [0.025, 0.975]
- Mean prediction: ~0.50 (well-calibrated)

---

## Competition Rules Note

This repository contains only the model code. The competition data files and `submission.csv` are excluded via `.gitignore` in compliance with [Kaggle's competition rules](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/rules), which prohibit redistribution of competition datasets.
