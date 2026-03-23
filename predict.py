"""
March Machine Learning Mania 2026 - Prediction Model (Men + Women)
Evaluation: Brier score (MSE on probabilities)

Features:
  - Margin-of-victory weighted Elo ratings
  - Massey ordinal rankings (median across 60+ systems, men's only)
  - Box score efficiency: offensive/defensive rating, eFG%, turnover rate
  - Logistic regression trained on 2010-2025 historical tourney games
  - Isotonic calibration
  - Gender-specific models (Men's TeamIDs < 3000, Women's >= 3000)

Usage:
  Place all competition CSV files in the same directory, then run:
      python predict.py
  Output: submission.csv
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

print("Loading data...")

m_seeds   = pd.read_csv("MNCAATourneySeeds.csv")
m_reg_c   = pd.read_csv("MRegularSeasonCompactResults.csv")
m_reg_d   = pd.read_csv("MRegularSeasonDetailedResults.csv")
m_tourney = pd.read_csv("MNCAATourneyCompactResults.csv")
m_massey  = pd.read_csv("MMasseyOrdinals.csv")

w_seeds   = pd.read_csv("WNCAATourneySeeds.csv")
w_reg_c   = pd.read_csv("WRegularSeasonCompactResults.csv")
w_reg_d   = pd.read_csv("WRegularSeasonDetailedResults.csv")
w_tourney = pd.read_csv("WNCAATourneyCompactResults.csv")

submission = pd.read_csv("SampleSubmissionStage2.csv")

np.random.seed(42)

# ── Helpers ────────────────────────────────────────────────────────────────────
def seed_to_num(s):
    """Extract numeric seed (1-16) from strings like 'W01', 'Z13a'."""
    return int(''.join(filter(str.isdigit, s)))

def elo_expected(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))

def build_seed_map(seeds_df):
    df = seeds_df.copy()
    df["SeedNum"] = df["Seed"].apply(seed_to_num)
    return df.set_index(["Season", "TeamID"])["SeedNum"].to_dict()

# ── Margin-of-victory Elo ──────────────────────────────────────────────────────
def build_elo(reg_c, tourney_c, K=20, HOME_ADV=100):
    """
    Elo with a margin-of-victory multiplier so blowouts update ratings more
    than close wins, following the FiveThirtyEight NBA Elo approach.
    """
    elo = defaultdict(lambda: 1500.0)
    all_games = pd.concat([
        reg_c[["Season","DayNum","WTeamID","LTeamID","WLoc","WScore","LScore"]],
        tourney_c[["Season","DayNum","WTeamID","LTeamID","WLoc","WScore","LScore"]]
    ]).sort_values(["Season","DayNum"])
    for _, row in all_games.iterrows():
        wt, lt = int(row["WTeamID"]), int(row["LTeamID"])
        margin = row["WScore"] - row["LScore"]
        elo_diff = abs(elo[wt] - elo[lt])
        # Log-dampened MoV multiplier (capped at 3x)
        mov = np.log(max(margin, 1) + 1) * (2.2 / (elo_diff * 0.001 + 2.2))
        mov = min(mov, 3.0)
        loc = row["WLoc"]
        ra = elo[wt] + (HOME_ADV if loc=="H" else -HOME_ADV if loc=="A" else 0)
        e  = elo_expected(ra, elo[lt])
        elo[wt] += K * mov * (1 - e)
        elo[lt] += K * mov * (0 - (1 - e))
    return elo

# ── Box score efficiency ───────────────────────────────────────────────────────
def build_efficiency(reg_d, recent_seasons):
    """
    Per-team offensive/defensive efficiency and shooting metrics
    from the last 3 seasons of detailed box score data.
    """
    stats = defaultdict(lambda: {"po":0,"pts_o":0,"pd":0,"pts_d":0,
                                  "fgm":0,"fga":0,"fgm3":0,"to":0,"g":0})
    for _, r in reg_d[reg_d["Season"].isin(recent_seasons)].iterrows():
        wt, lt = int(r["WTeamID"]), int(r["LTeamID"])
        # Possession estimate: FGA - OReb + TO + 0.44*FTA
        wp = r["WFGA"] - r["WOR"] + r["WTO"] + 0.44*r["WFTA"]
        lp = r["LFGA"] - r["LOR"] + r["LTO"] + 0.44*r["LFTA"]
        p  = (wp + lp) / 2
        for tid, pts_o, pts_d, fgm, fga, fgm3, to in [
            (wt, r["WScore"], r["LScore"], r["WFGM"], r["WFGA"], r["WFGM3"], r["WTO"]),
            (lt, r["LScore"], r["WScore"], r["LFGM"], r["LFGA"], r["LFGM3"], r["LTO"]),
        ]:
            s = stats[tid]
            s["po"]+=p; s["pts_o"]+=pts_o; s["pd"]+=p; s["pts_d"]+=pts_d
            s["fgm"]+=fgm; s["fga"]+=fga; s["fgm3"]+=fgm3; s["to"]+=to; s["g"]+=1
    eff = {}
    for tid, s in stats.items():
        if s["po"] > 0:
            eff[tid] = {
                "ortg": 100*s["pts_o"]/s["po"],          # points per 100 possessions (offense)
                "drtg": 100*s["pts_d"]/s["pd"],          # points allowed per 100 possessions (defense)
                "efg":  (s["fgm"]+0.5*s["fgm3"])/max(s["fga"],1),  # effective FG%
                "to_r": s["to"]/max(s["po"],1),          # turnover rate
            }
    return eff

# ── Massey ordinals (men's only) ───────────────────────────────────────────────
def build_massey(massey_df, season):
    """
    Median ordinal rank across all available rating systems for the given season,
    using the most recent 2 weeks of rankings before the tournament.
    Lower rank = better team. We flip the sign in featurize() so positive = better.
    """
    s = massey_df[massey_df["Season"] == season]
    if s.empty:
        return {}
    latest = s["RankingDayNum"].max()
    s = s[s["RankingDayNum"] >= latest - 14]
    return s.groupby("TeamID")["OrdinalRank"].median().to_dict()

# ── Feature vector ─────────────────────────────────────────────────────────────
def featurize(t1, t2, season, elo, eff, smap, mmap):
    """
    7-feature vector from t1's perspective (positive = t1 advantage):
      [elo_diff, seed_diff, ortg_diff, drtg_diff, efg_diff, to_diff, massey_diff]
    """
    elo_diff  = elo[t1] - elo[t2]
    s1 = smap.get((season, t1)); s2 = smap.get((season, t2))
    seed_diff = (s2 - s1) if (s1 and s2) else 0.0   # positive = t1 is better seed
    e1 = eff.get(t1, {}); e2 = eff.get(t2, {})
    ortg_diff = e1.get("ortg",100) - e2.get("ortg",100)
    drtg_diff = e2.get("drtg",100) - e1.get("drtg",100)  # flipped: lower drtg is better
    efg_diff  = e1.get("efg",0.5)  - e2.get("efg",0.5)
    to_diff   = e2.get("to_r",0.2) - e1.get("to_r",0.2)  # flipped: lower TO rate is better
    m1 = mmap.get(t1, 200); m2 = mmap.get(t2, 200)
    massey_diff = m2 - m1   # flipped: lower rank = better, so positive = t1 better
    return [elo_diff, seed_diff, ortg_diff, drtg_diff, efg_diff, to_diff, massey_diff]

# ── Train logistic regression + isotonic calibration ──────────────────────────
def build_model(seeds_df, reg_c, reg_d, tourney_df, massey_df, label, recent):
    print(f"Building {label} model...")
    smap = build_seed_map(seeds_df)
    elo  = build_elo(reg_c, tourney_df)
    eff  = build_efficiency(reg_d, recent)
    mmap = build_massey(massey_df, 2026) if massey_df is not None else {}

    print(f"  Training {label} logistic regression on 2010-2025 tourney games...")
    X_train, y_train = [], []
    train_games = tourney_df[(tourney_df["Season"] >= 2010) & (tourney_df["Season"] <= 2025)]
    for _, row in train_games.iterrows():
        wt, lt = int(row["WTeamID"]), int(row["LTeamID"])
        s = int(row["Season"])
        X_train.append(featurize(wt, lt, s, elo, eff, smap, mmap)); y_train.append(1)
        X_train.append(featurize(lt, wt, s, elo, eff, smap, mmap)); y_train.append(0)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    lr = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
    lr.fit(Xs, y_train)

    raw_probs = lr.predict_proba(Xs)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_probs, y_train)

    print(f"  {label}: trained on {len(train_games)} games")
    return scaler, lr, iso, elo, eff, smap, mmap

# ── Build both models ──────────────────────────────────────────────────────────
m_scaler, m_lr, m_iso, m_elo, m_eff, m_smap, m_mmap = build_model(
    m_seeds, m_reg_c, m_reg_d, m_tourney, m_massey, "Men's", [2024,2025,2026]
)
w_scaler, w_lr, w_iso, w_elo, w_eff, w_smap, w_mmap = build_model(
    w_seeds, w_reg_c, w_reg_d, w_tourney, None, "Women's", [2024,2025,2026]
)

# ── Generate predictions ───────────────────────────────────────────────────────
print("Generating predictions...")

def predict_one(t1, t2, season, scaler, lr, iso, elo, eff, smap, mmap):
    feats = featurize(t1, t2, season, elo, eff, smap, mmap)
    X = scaler.transform([feats])
    raw = lr.predict_proba(X)[0, 1]
    cal = float(iso.predict([raw])[0])
    return float(np.clip(cal, 0.025, 0.975))

preds = []
for _, row in submission.iterrows():
    parts  = row["ID"].split("_")
    season = int(parts[0])
    t1, t2 = int(parts[1]), int(parts[2])
    if t1 < 3000:
        p = predict_one(t1, t2, season, m_scaler, m_lr, m_iso, m_elo, m_eff, m_smap, m_mmap)
    else:
        p = predict_one(t1, t2, season, w_scaler, w_lr, w_iso, w_elo, w_eff, w_smap, w_mmap)
    preds.append(p)

submission["Pred"] = preds
submission.to_csv("submission.csv", index=False)

print(f"\nDone! {len(submission)} predictions written to submission.csv")
print(f"Stats — min: {submission['Pred'].min():.4f}, max: {submission['Pred'].max():.4f}, mean: {submission['Pred'].mean():.4f}")
mens   = submission[submission["ID"].apply(lambda x: int(x.split("_")[1]) < 3000)]
womens = submission[submission["ID"].apply(lambda x: int(x.split("_")[1]) >= 3000)]
print(f"Men's: {len(mens)} predictions, Women's: {len(womens)} predictions")
