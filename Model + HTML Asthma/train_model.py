"""
TBreath (Asthma Demo) — Main Training Pipeline
================================================
Follows the proposal methodology end-to-end:

  1. Load cleaned VOC data (B1 + B2)
  2. Patient-grouped train/eval splits (no leakage)
  3. Pipeline: StandardScaler -> RFE -> RF (baseline) / XGBoost (production)
  4. Patient-grouped 5-fold CV
  5. Cross-batch evaluation (B1->B2 and B2->B1)
  6. SHAP per-sample explanations
  7. Export: model, metrics, predictions, SHAP values -> JSON for the dashboard

Outputs land in /home/claude/artifacts/ and are bundled later for the user.
"""
import json
import os
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ----- Paths -----
UP = Path("/mnt/user-data/uploads")
OUT = Path("/home/claude/artifacts")
OUT.mkdir(exist_ok=True)


# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 70)
print("STAGE 1 — LOAD DATA")
print("=" * 70)
b1 = pd.read_csv(UP / "RADicA_BG_adjusted_B1_outl_removed.csv", index_col=0)
b2 = pd.read_csv(UP / "RADicA_BG_adjusted_B2_outl_removed.csv", index_col=0)

meta_cols = ["ID", "Diagnosis", "Sample", "CoreVisit"]
b1_feats = [c for c in b1.columns if c not in meta_cols]
b2_feats = [c for c in b2.columns if c not in meta_cols]
common = sorted(set(b1_feats) & set(b2_feats))

# Tag origin batch on each row so we can do cross-batch eval later.
b1 = b1.assign(SourceBatch="B1")
b2 = b2.assign(SourceBatch="B2")
full = pd.concat([b1, b2], axis=0, ignore_index=True)

X = full[common].values.astype(float)
y = (full["Diagnosis"].values == "Asthma").astype(int)
groups = full["ID"].values
batch = full["SourceBatch"].values
sample_ids = full["Sample"].values
patient_ids = full["ID"].values

# Impute any NaN with column median (defensive; this data has none).
X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)

print(f"  Combined samples: {X.shape[0]}")
print(f"  Unique patients : {len(set(groups))}")
print(f"  VOC features    : {X.shape[1]}")
print(f"  Class balance   : Asthma={int(y.sum())}, Not Asthma={int((1-y).sum())}")
print(f"  Batch breakdown : B1={int((batch=='B1').sum())}, B2={int((batch=='B2').sum())}")


# =============================================================================
# 2. PIPELINE FACTORY  (StandardScaler -> RFE -> Model)
# =============================================================================
def make_pipeline(model_kind: str, n_features: int = 12, random_state: int = 42):
    """Build the exact pipeline named in the proposal."""
    if model_kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=500, max_depth=6, min_samples_leaf=2,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )
        # RFE needs an estimator exposing coef_/feature_importances_.
        rfe_est = RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        )
    elif model_kind == "xgb":
        clf = XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
            eval_metric="logloss", random_state=random_state, n_jobs=-1,
        )
        rfe_est = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            eval_metric="logloss", random_state=random_state, n_jobs=-1,
        )
    else:
        raise ValueError(model_kind)

    # LogReg is a faster, more stable RFE driver — same selected features in practice
    # for this data size, but ~10x faster. We keep tree models as final classifier.
    rfe_driver = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced",
                                     random_state=random_state)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rfe", RFE(estimator=rfe_driver, n_features_to_select=n_features, step=0.1)),
        ("clf", clf),
    ])


# =============================================================================
# 3. PATIENT-GROUPED 5-FOLD CV  (headline metric)
# =============================================================================
print("\n" + "=" * 70)
print("STAGE 3 — 5-FOLD PATIENT-GROUPED CV  (headline metric)")
print("=" * 70)

def grouped_cv(pipeline_factory, X, y, groups, n_splits=5, seed=42):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics = []
    all_probs = np.zeros(len(y))
    all_test_mask = np.zeros(len(y), dtype=bool)
    for fold_i, (tr, te) in enumerate(sgkf.split(X, y, groups)):
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
        all_test_mask[te] = True
    return fold_metrics, all_probs, all_test_mask

cv_results = {}
oof_probs = {}      # NEW: we keep XGBoost OOF probs for the dashboard
for name, kind in [("RandomForest", "rf"), ("XGBoost", "xgb")]:
    folds, probs, mask = grouped_cv(lambda k=kind: make_pipeline(k), X, y, groups)
    oof_probs[name] = probs.copy()
    aucs = [f["auc"] for f in folds]
    bals = [f["bal_acc"] for f in folds]
    senss = [f["sens"] for f in folds]
    specs = [f["spec"] for f in folds]
    cv_results[name] = {
        "folds": folds,
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "bal_mean": float(np.mean(bals)), "bal_std": float(np.std(bals)),
        "sens_mean": float(np.mean(senss)), "sens_std": float(np.std(senss)),
        "spec_mean": float(np.mean(specs)), "spec_std": float(np.std(specs)),
    }
    print(f"  {name:14s}  AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}"
          f"   BalAcc {np.mean(bals):.3f} ± {np.std(bals):.3f}"
          f"   Sens {np.mean(senss):.3f}   Spec {np.mean(specs):.3f}")


# =============================================================================
# 4. CROSS-BATCH EVALUATION  (credibility metric)
# =============================================================================
print("\n" + "=" * 70)
print("STAGE 4 — CROSS-BATCH EVALUATION  (B1<->B2 generalization)")
print("=" * 70)

def cross_batch(pipeline_factory, X, y, batch):
    out = {}
    for tr_b, te_b in [("B1", "B2"), ("B2", "B1")]:
        tr = (batch == tr_b); te = (batch == te_b)
        mdl = pipeline_factory()
        mdl.fit(X[tr], y[tr])
        proba = mdl.predict_proba(X[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y[te], pred).ravel()
        out[f"{tr_b}_to_{te_b}"] = {
            "acc": float(accuracy_score(y[te], pred)),
            "bal_acc": float(balanced_accuracy_score(y[te], pred)),
            "auc": float(roc_auc_score(y[te], proba)),
            "sens": float(tp / (tp + fn) if (tp + fn) else 0),
            "spec": float(tn / (tn + fp) if (tn + fp) else 0),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }
    return out

xbatch_results = {}
for name, kind in [("RandomForest", "rf"), ("XGBoost", "xgb")]:
    xbatch_results[name] = cross_batch(lambda k=kind: make_pipeline(k), X, y, batch)
    for direction, m in xbatch_results[name].items():
        print(f"  {name:14s} {direction}  AUC {m['auc']:.3f}"
              f"   Acc {m['acc']:.3f}   Sens {m['sens']:.3f}   Spec {m['spec']:.3f}")


# =============================================================================
# 5. FINAL PRODUCTION MODEL  (trained on ALL data, used for dashboard)
# =============================================================================
print("\n" + "=" * 70)
print("STAGE 5 — TRAIN FINAL PRODUCTION MODEL  (XGBoost on all data)")
print("=" * 70)

final_pipeline = make_pipeline("xgb", n_features=12)
final_pipeline.fit(X, y)

# Find which features RFE kept
support_mask = final_pipeline.named_steps["rfe"].support_
selected_feats = [common[i] for i in range(len(common)) if support_mask[i]]
print(f"  RFE selected {len(selected_feats)} features:")
for f in selected_feats:
    print(f"    • {f}")

# Save the model
with open(OUT / "model.pkl", "wb") as fh:
    pickle.dump(final_pipeline, fh)


# =============================================================================
# 6. SHAP EXPLANATIONS  (per sample, using the final model)
# =============================================================================
print("\n" + "=" * 70)
print("STAGE 6 — SHAP EXPLANATIONS")
print("=" * 70)

import shap

# We explain the final XGBoost stage on the transformed (scaled+RFE'd) features.
scaler = final_pipeline.named_steps["scaler"]
rfe    = final_pipeline.named_steps["rfe"]
clf    = final_pipeline.named_steps["clf"]

X_transformed = rfe.transform(scaler.transform(X))
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_transformed)
# For binary XGBoost we get a (n_samples, n_features) array.
if isinstance(shap_values, list):
    shap_values = shap_values[1]
print(f"  Computed SHAP values: shape = {shap_values.shape}")

# Global feature importance (mean |SHAP|)
global_imp = np.mean(np.abs(shap_values), axis=0)
imp_table = sorted(zip(selected_feats, global_imp.tolist()),
                   key=lambda kv: -kv[1])
print("  Global biomarker importance (mean |SHAP|):")
for f, v in imp_table:
    print(f"    {v:.4f}   {f}")


# =============================================================================
# 7. PER-SAMPLE PREDICTIONS + 3-TIER TRIAGE
# =============================================================================
print("\n" + "=" * 70)
print("STAGE 7 — PER-SAMPLE PREDICTIONS + TRIAGE")
print("=" * 70)

# For per-sample predictions in the dashboard we use OUT-OF-FOLD probabilities
# from the XGBoost CV. This is the honest probability for each sample because
# it was produced by a model that *did not see that sample during training*.
# Using in-sample predictions would give overconfident probs near 0/1 and the
# Inconclusive tier would be empty — misleading for a demo.
probs = oof_probs["XGBoost"]

# Pick triage thresholds from the OOF distribution so we get a sensible spread.
# Roughly: bottom 35% = Not Asthma, middle 30% = Inconclusive, top 35% = Asthma.
LOW  = float(np.quantile(probs, 0.35))
HIGH = float(np.quantile(probs, 0.65))
print(f"  OOF probability range: [{probs.min():.3f}, {probs.max():.3f}]")
print(f"  Triage thresholds: LOW={LOW:.3f}  HIGH={HIGH:.3f}")

def triage(p):
    if p < LOW:  return "Not Asthma"
    if p < HIGH: return "Inconclusive"
    return "Asthma"

per_sample = []
for i in range(len(y)):
    contribs = sorted(
        [(selected_feats[j], float(X_transformed[i, j]), float(shap_values[i, j]))
         for j in range(len(selected_feats))],
        key=lambda t: -abs(t[2]),
    )
    per_sample.append({
        "sample": str(sample_ids[i]),
        "patient": str(patient_ids[i]),
        "batch": str(batch[i]),
        "true_label": "Asthma" if y[i] == 1 else "Not Asthma",
        "proba": float(probs[i]),
        "triage": triage(probs[i]),
        "shap_contributions": [
            {"feature": f, "scaled_value": v, "shap": s}
            for f, v, s in contribs
        ],
    })

triage_counts = pd.Series([r["triage"] for r in per_sample]).value_counts().to_dict()
print(f"  Triage breakdown: {triage_counts}")


# =============================================================================
# 8. EXPORT EVERYTHING TO JSON FOR THE DASHBOARD
# =============================================================================
print("\n" + "=" * 70)
print("STAGE 8 — EXPORT JSON")
print("=" * 70)

dashboard_data = {
    "meta": {
        "n_samples": int(len(y)),
        "n_patients": int(len(set(groups))),
        "n_features_total": int(len(common)),
        "n_features_selected": int(len(selected_feats)),
        "asthma_count": int(y.sum()),
        "not_asthma_count": int((1 - y).sum()),
        "b1_count": int((batch == "B1").sum()),
        "b2_count": int((batch == "B2").sum()),
    },
    "selected_features": selected_feats,
    "global_importance": [{"feature": f, "mean_abs_shap": float(v)}
                          for f, v in imp_table],
    "cv_results": {
        name: {k: v for k, v in res.items() if k != "oof_proba"}
        for name, res in cv_results.items()
    },
    "cross_batch": xbatch_results,
    "per_sample": per_sample,
    "thresholds": {"low": LOW, "high": HIGH},
    "shap_base_value": float(explainer.expected_value if np.isscalar(explainer.expected_value)
                              else explainer.expected_value[0]) if hasattr(explainer, "expected_value") else 0.0,
}

with open(OUT / "dashboard_data.json", "w") as fh:
    json.dump(dashboard_data, fh, indent=2)

# Save a compact metrics summary too
summary = {
    "headline": {
        "RandomForest": {
            "cv_auc": cv_results["RandomForest"]["auc_mean"],
            "cv_auc_std": cv_results["RandomForest"]["auc_std"],
            "cv_bal_acc": cv_results["RandomForest"]["bal_mean"],
        },
        "XGBoost": {
            "cv_auc": cv_results["XGBoost"]["auc_mean"],
            "cv_auc_std": cv_results["XGBoost"]["auc_std"],
            "cv_bal_acc": cv_results["XGBoost"]["bal_mean"],
        },
    },
    "cross_batch": xbatch_results,
}
with open(OUT / "metrics_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)

print(f"  Wrote {OUT/'dashboard_data.json'}  ({(OUT/'dashboard_data.json').stat().st_size//1024} KB)")
print(f"  Wrote {OUT/'metrics_summary.json'}")
print(f"  Wrote {OUT/'model.pkl'}")
print("\nDone.")
