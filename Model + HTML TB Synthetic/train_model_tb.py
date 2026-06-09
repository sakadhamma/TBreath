"""
TBreath — Training Pipeline for synthetic microGC TB dataset.

Mirrors train_model.py (asthma) but adapted for:
  - Excel input with RT/Area/Height/SNR per VOC (32 features × 8 VOCs)
  - Balanced 300 samples (150 TB / 150 controls)
  - No patient IDs (synthetic data) → StratifiedKFold instead of grouped
  - Batch column repurposed for cross-batch evaluation (5-batch leave-one-out)
  - TB-specific output labels and clinical recommendations

NOTE: dataset is SYNTHETIC. High AUC here is not clinical evidence — it shows
the pipeline classifies cleanly-separable data correctly.
"""
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

UP = Path("/mnt/user-data/uploads")
OUT = Path("/home/claude/artifacts_tb")
OUT.mkdir(exist_ok=True)


# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 70); print("STAGE 1 — LOAD"); print("=" * 70)

df = pd.read_excel(UP / "synthetic_microgc_rawstyle_tb_dataset.xlsx")
feature_cols = [c for c in df.columns if c not in ("Sample_ID", "TB_Status", "Batch")]

X = df[feature_cols].values.astype(float)
y = df["TB_Status"].values.astype(int)              # 1=TB, 0=Control
batch = df["Batch"].values.astype(int)
sample_ids = df["Sample_ID"].astype(str).values

print(f"  Samples : {X.shape[0]}")
print(f"  Features: {X.shape[1]}  (across {sum(c.endswith('_RT_s') for c in feature_cols)} VOCs)")
print(f"  Class   : TB={int(y.sum())}, Control={int((1-y).sum())}")
print(f"  Batches : {dict(zip(*np.unique(batch, return_counts=True)))}")


# =============================================================================
# 2. PIPELINE FACTORY (StandardScaler -> RFE -> Model)
# =============================================================================
def make_pipeline(model_kind: str, n_features: int = 10, random_state: int = 42):
    if model_kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=500, max_depth=6, min_samples_leaf=2,
            class_weight="balanced", random_state=random_state, n_jobs=-1)
    elif model_kind == "xgb":
        clf = XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
            eval_metric="logloss", random_state=random_state, n_jobs=-1)
    else:
        raise ValueError(model_kind)
    rfe_driver = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced",
                                     random_state=random_state)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rfe", RFE(estimator=rfe_driver, n_features_to_select=n_features, step=0.1)),
        ("clf", clf),
    ])


# =============================================================================
# 3. 5-FOLD STRATIFIED CV (headline metric)
# =============================================================================
print("\n" + "=" * 70); print("STAGE 3 — 5-FOLD STRATIFIED CV"); print("=" * 70)

def stratified_cv(pipeline_factory, X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics = []
    all_probs = np.zeros(len(y))
    for fold_i, (tr, te) in enumerate(skf.split(X, y)):
        mdl = pipeline_factory()
        mdl.fit(X[tr], y[tr])
        proba = mdl.predict_proba(X[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y[te], pred).ravel()
        fold_metrics.append({
            "fold": fold_i + 1, "n_test": len(te),
            "acc": accuracy_score(y[te], pred),
            "bal_acc": balanced_accuracy_score(y[te], pred),
            "auc": roc_auc_score(y[te], proba),
            "sens": tp / (tp + fn) if (tp + fn) else 0.0,
            "spec": tn / (tn + fp) if (tn + fp) else 0.0,
        })
        all_probs[te] = proba
    return fold_metrics, all_probs

cv_results = {}
oof_probs = {}
for name, kind in [("RandomForest", "rf"), ("XGBoost", "xgb")]:
    folds, probs = stratified_cv(lambda k=kind: make_pipeline(k), X, y)
    oof_probs[name] = probs.copy()
    aucs = [f["auc"] for f in folds]; bals = [f["bal_acc"] for f in folds]
    senss = [f["sens"] for f in folds]; specs = [f["spec"] for f in folds]
    cv_results[name] = {
        "folds": folds,
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "bal_mean": float(np.mean(bals)), "bal_std": float(np.std(bals)),
        "sens_mean": float(np.mean(senss)), "sens_std": float(np.std(senss)),
        "spec_mean": float(np.mean(specs)), "spec_std": float(np.std(specs)),
    }
    print(f"  {name:14s}  AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}"
          f"   BalAcc {np.mean(bals):.3f}"
          f"   Sens {np.mean(senss):.3f}   Spec {np.mean(specs):.3f}")


# =============================================================================
# 4. LEAVE-ONE-BATCH-OUT EVALUATION (credibility metric)
# =============================================================================
print("\n" + "=" * 70); print("STAGE 4 — LEAVE-ONE-BATCH-OUT"); print("=" * 70)

def lobo(pipeline_factory, X, y, batch):
    out = {}
    for held in sorted(np.unique(batch)):
        tr = (batch != held); te = (batch == held)
        mdl = pipeline_factory()
        mdl.fit(X[tr], y[tr])
        proba = mdl.predict_proba(X[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y[te], pred).ravel()
        out[f"holdout_batch_{held}"] = {
            "n_test": int(te.sum()),
            "auc": float(roc_auc_score(y[te], proba)),
            "acc": float(accuracy_score(y[te], pred)),
            "sens": float(tp / (tp + fn) if (tp + fn) else 0),
            "spec": float(tn / (tn + fp) if (tn + fp) else 0),
        }
    return out

lobo_results = {}
for name, kind in [("RandomForest", "rf"), ("XGBoost", "xgb")]:
    lobo_results[name] = lobo(lambda k=kind: make_pipeline(k), X, y, batch)
    aucs = [m["auc"] for m in lobo_results[name].values()]
    print(f"  {name:14s}  LOBO AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")


# =============================================================================
# 5. FINAL MODEL (full data, used by dashboard for SHAP)
# =============================================================================
print("\n" + "=" * 70); print("STAGE 5 — TRAIN FINAL MODEL"); print("=" * 70)

final_pipeline = make_pipeline("xgb", n_features=10)
final_pipeline.fit(X, y)
support_mask = final_pipeline.named_steps["rfe"].support_
selected_feats = [feature_cols[i] for i in range(len(feature_cols)) if support_mask[i]]
print(f"  RFE selected {len(selected_feats)} features:")
for f in selected_feats: print(f"    • {f}")

with open(OUT / "model.pkl", "wb") as fh:
    pickle.dump(final_pipeline, fh)


# =============================================================================
# 6. SHAP
# =============================================================================
print("\n" + "=" * 70); print("STAGE 6 — SHAP"); print("=" * 70)
import shap
scaler = final_pipeline.named_steps["scaler"]
rfe    = final_pipeline.named_steps["rfe"]
clf    = final_pipeline.named_steps["clf"]
X_transformed = rfe.transform(scaler.transform(X))
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_transformed)
if isinstance(shap_values, list): shap_values = shap_values[1]

global_imp = np.mean(np.abs(shap_values), axis=0)
imp_table = sorted(zip(selected_feats, global_imp.tolist()), key=lambda kv: -kv[1])
print("  Global biomarker importance (mean |SHAP|):")
for f, v in imp_table: print(f"    {v:.4f}   {f}")


# =============================================================================
# 7. PER-SAMPLE PREDICTIONS + TRIAGE (TB labels per proposal)
# =============================================================================
print("\n" + "=" * 70); print("STAGE 7 — PER-SAMPLE + TRIAGE"); print("=" * 70)

probs = oof_probs["XGBoost"]
# Fixed clinical thresholds (not percentile-based as in the asthma version).
# On synthetic data the model is extremely confident in almost every prediction;
# using percentiles would force ~30% of samples into "Suspected" even when the
# model is highly confident about them. Fixed thresholds correctly reflect what
# would happen in deployment: confident calls stay confident, only genuinely
# borderline probabilities (0.20-0.80) trigger the safety net.
LOW, HIGH = 0.20, 0.80
print(f"  OOF range: [{probs.min():.3f}, {probs.max():.3f}]")
print(f"  Thresholds (fixed): LOW={LOW:.2f}  HIGH={HIGH:.2f}")

def triage(p):
    if p < LOW:  return "Negative"
    if p < HIGH: return "Suspected"
    return "Positive"

per_sample = []
for i in range(len(y)):
    contribs = sorted(
        [(selected_feats[j], float(X_transformed[i, j]), float(shap_values[i, j]))
         for j in range(len(selected_feats))],
        key=lambda t: -abs(t[2]),
    )
    per_sample.append({
        "sample": str(sample_ids[i]),
        "batch": int(batch[i]),
        "true_label": "TB Positive" if y[i] == 1 else "Control",
        "proba": float(probs[i]),
        "triage": triage(probs[i]),
        "shap_contributions": [
            {"feature": f, "scaled_value": v, "shap": s} for f, v, s in contribs
        ],
    })

tri_counts = pd.Series([r["triage"] for r in per_sample]).value_counts().to_dict()
print(f"  Triage breakdown: {tri_counts}")


# =============================================================================
# 8. EXPORT JSON
# =============================================================================
print("\n" + "=" * 70); print("STAGE 8 — EXPORT JSON"); print("=" * 70)

n_vocs = sum(c.endswith("_RT_s") for c in feature_cols)
dashboard_data = {
    "meta": {
        "is_synthetic": True,
        "dataset_name": "Synthetic microGC TB dataset",
        "n_samples": int(len(y)),
        "n_features_total": int(len(feature_cols)),
        "n_features_selected": int(len(selected_feats)),
        "n_vocs": int(n_vocs),
        "tb_positive_count": int(y.sum()),
        "control_count": int((1 - y).sum()),
        "n_batches": int(len(np.unique(batch))),
    },
    "selected_features": selected_feats,
    "global_importance": [{"feature": f, "mean_abs_shap": float(v)} for f, v in imp_table],
    "cv_results": cv_results,
    "leave_one_batch_out": lobo_results,
    "per_sample": per_sample,
    "thresholds": {"low": LOW, "high": HIGH},
}

with open(OUT / "dashboard_data.json", "w") as fh:
    json.dump(dashboard_data, fh, indent=2)

print(f"  Wrote {OUT/'dashboard_data.json'}  "
      f"({(OUT/'dashboard_data.json').stat().st_size//1024} KB)")
print(f"  Wrote {OUT/'model.pkl'}")
print("\nDone.")
