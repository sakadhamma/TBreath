"""
Benchmark — compare classifiers on the RADicA asthma dataset.

All classifiers run through the SAME preprocessing pipeline (StandardScaler -> RFE)
so the comparison isolates the effect of the model choice. Patient-grouped
5-fold cross-validation is used throughout (no leakage between train/test).

Run:  python src/benchmark.py
Output: artifacts/benchmark_results.json + console table
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
UP   = Path("/mnt/user-data/uploads")
OUT  = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)


# ----- Load data -----
b1 = pd.read_csv(UP / "RADicA_BG_adjusted_B1_outl_removed.csv", index_col=0)
b2 = pd.read_csv(UP / "RADicA_BG_adjusted_B2_outl_removed.csv", index_col=0)
meta_cols = ["ID", "Diagnosis", "Sample", "CoreVisit"]
common = sorted(set(c for c in b1.columns if c not in meta_cols) &
                set(c for c in b2.columns if c not in meta_cols))
full = pd.concat([b1, b2], ignore_index=True)
X = full[common].values.astype(float)
y = (full["Diagnosis"].values == "Asthma").astype(int)
groups = full["ID"].values
X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)

print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, "
      f"{len(set(groups))} unique patients")
print(f"Class balance: Asthma={int(y.sum())}, Not Asthma={int((1-y).sum())}\n")


# ----- Models to benchmark -----
RANDOM_STATE = 42
def rfe_driver():
    return LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced",
                              random_state=RANDOM_STATE)

def wrap(clf):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rfe", RFE(estimator=rfe_driver(), n_features_to_select=12, step=0.1)),
        ("clf", clf),
    ])

models = {
    "Logistic Regression (L2)": wrap(LogisticRegression(
        max_iter=2000, C=0.5, class_weight="balanced", random_state=RANDOM_STATE)),
    "Logistic Regression (L1)": wrap(LogisticRegression(
        max_iter=2000, C=0.5, penalty="l1", solver="liblinear",
        class_weight="balanced", random_state=RANDOM_STATE)),
    "k-Nearest Neighbours (k=7)": wrap(KNeighborsClassifier(n_neighbors=7)),
    "Gaussian Naive Bayes": wrap(GaussianNB()),
    "Quadratic Discriminant": wrap(QuadraticDiscriminantAnalysis(reg_param=0.1)),
    "SVM (RBF kernel)": wrap(SVC(C=1.0, kernel="rbf", probability=True,
                                  class_weight="balanced", random_state=RANDOM_STATE)),
    "Random Forest (baseline)": wrap(RandomForestClassifier(
        n_estimators=500, max_depth=6, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
    "XGBoost (production)": wrap(XGBClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)),
}


# ----- Evaluate -----
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = []
print(f"{'Model':<28} {'AUC':>14} {'BalAcc':>14} {'Sens':>10} {'Spec':>10} {'Time(s)':>10}")
print("-" * 90)

for name, base in models.items():
    aucs, bals, senss, specs = [], [], [], []
    t0 = time.perf_counter()
    for tr, te in sgkf.split(X, y, groups):
        mdl = clone(base)
        mdl.fit(X[tr], y[tr])
        proba = mdl.predict_proba(X[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y[te], pred).ravel()
        aucs.append(roc_auc_score(y[te], proba))
        bals.append(balanced_accuracy_score(y[te], pred))
        senss.append(tp / (tp + fn) if (tp + fn) else 0.0)
        specs.append(tn / (tn + fp) if (tn + fp) else 0.0)
    elapsed = time.perf_counter() - t0
    row = {
        "model": name,
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "bal_mean": float(np.mean(bals)), "bal_std": float(np.std(bals)),
        "sens_mean": float(np.mean(senss)), "spec_mean": float(np.mean(specs)),
        "cv_time_sec": float(elapsed),
    }
    results.append(row)
    print(f"{name:<28} "
          f"{row['auc_mean']:.3f} ± {row['auc_std']:.2f}  "
          f"{row['bal_mean']:.3f} ± {row['bal_std']:.2f}  "
          f"{row['sens_mean']:>8.3f}  {row['spec_mean']:>8.3f}  "
          f"{elapsed:>8.2f}")

# Save
with open(OUT / "benchmark_results.json", "w") as fh:
    json.dump(results, fh, indent=2)
print(f"\nSaved -> {OUT/'benchmark_results.json'}")

# Pretty markdown table
md = ["| Model | AUC | Balanced Accuracy | Sensitivity | Specificity |",
      "|---|---|---|---|---|"]
for r in sorted(results, key=lambda x: -x["auc_mean"]):
    md.append(f"| {r['model']} "
              f"| {r['auc_mean']:.3f} ± {r['auc_std']:.2f} "
              f"| {r['bal_mean']:.3f} "
              f"| {r['sens_mean']:.3f} "
              f"| {r['spec_mean']:.3f} |")
(OUT / "benchmark_table.md").write_text("\n".join(md))
print(f"Saved -> {OUT/'benchmark_table.md'}")
