# TBreath

**AI-assisted breathomics for early tuberculosis screening — Proof of Concept.**

TBreath is an end-to-end screening prototype that classifies breath samples from a microGC + PID hardware front-end. It is currently demonstrated on a publicly available **asthma** dataset (RADicA) because no public TB breath-VOC dataset exists. The pipeline, model, and dashboard transfer directly once TB patient data is collected.

> ⚠️ **Status: Proof of Concept.** The classifier has never seen TB patient data. The asthma dataset is used as a methodological proxy to validate the pipeline end-to-end. Clinical TB performance claims require collection of real patient samples.

---

## Architecture at a glance

```
   ┌──────────────┐    ┌────────────────────┐    ┌──────────────────┐
   │  microGC+PID │    │   Edge device      │    │  Cloud / Local   │
   │   Hardware   │───>│  (Raspberry Pi /   │───>│   Inference      │
   │              │    │   ESP32)           │    │                  │
   └──────────────┘    │ ┌────────────────┐ │    │ ┌──────────────┐ │
                       │ │ALS baseline    │ │    │ │ StandardScaler│ │
                       │ │SavGol smooth   │ │    │ │ + RFE         │ │
                       │ │peak detection  │ │    │ │ + XGBoost     │ │
                       │ └────────────────┘ │    │ │ + SHAP        │ │
                       └────────────────────┘    │ └──────────────┘ │
                                                 │       │          │
                                                 │       ▼          │
                                                 │  Triage output:  │
                                                 │  Pos / Susp / Neg│
                                                 │  + confidence    │
                                                 │  + SHAP top VOCs │
                                                 └──────────────────┘
```

End-to-end latency per sample (measured): **~138 ms** (preprocessing 136 ms + inference ~1 ms with XGBoost).

---

## Repository structure

```
TBreath/
├── Model + HTML Asthma/                 ← Asthma PoC (RADicA dataset)
│   ├── artifacts/
│   │   ├── benchmark_results.json       ← classifier comparison metrics (8 models)
│   │   ├── dashboard_data.json          ← model outputs, per-sample predictions, SHAP
│   │   └── inference_timing.json        ← per-phase latency (ms)
│   ├── dashboard/
│   │   ├── TBreath_dashboard.html           ← asthma dashboard (open in any browser)
│   │   └── TBreath_dashboard-compacted.html ← minified version for edge/low-bandwidth
│   ├── src/
│   │   ├── benchmark.py                 ← compare 8 classifiers (LR, SVM, RF, XGB, kNN, NB, QDA)
│   │   ├── inference_timing.py          ← per-phase latency measurement
│   │   ├── preprocessing_demo.py        ← synthetic chromatogram + ALS + SavGol + peaks
│   │   └── train_model.py               ← main PoC training pipeline (StandardScaler + RFE + XGBoost + SHAP)
│   └── benchmark_table.md               ← human-readable benchmark summary table
│
└── Model + HTML TB Synthetic/           ← TB model (Figshare 29504333 synthetic data)
    ├── TBreath_TB_dashboard.html        ← TB dashboard (open in any browser)
    ├── dashboard_data_tb.json           ← model outputs, per-sample predictions, SHAP
    ├── train_model_tb.py                ← TB synthetic-data training pipeline
    ├── DataSources.md                   ← data source URLs and citations (no PII redistributed)
    ├── README.md
    └── requirements.txt                 ← all the requirements needed to run the program
```

> Note: `benchmark.py` and `inference_timing.py` currently exist only under `Model + HTML Asthma/src/` — there is no equivalent classifier benchmark or latency script yet for the TB synthetic pipeline.

---

## Quick start

### 1. Install dependencies
```bash
pip install -r "Model + HTML TB Synthetic/requirements.txt"
```

### 2. View dashboards
The HTML dashboards are fully self-contained — open them in any browser:
```bash
# macOS / Linux
open "Model + HTML Asthma/dashboard/TBreath_dashboard.html"        # asthma PoC
open "Model + HTML TB Synthetic/TBreath_TB_dashboard.html"         # TB synthetic
```
No backend required. All model outputs, SHAP values, and visualisations are embedded.

### 3. Reproduce the model from scratch
The training scripts assume the source datasets are at `/mnt/user-data/uploads/` (the original Anthropic container path) — adjust the `UP = Path(...)` line at the top of each script to point to wherever you've downloaded the raw files.

```bash
# Preprocessing demo (saves chromatogram PNG)
python "Model + HTML Asthma/src/preprocessing_demo.py"

# Asthma PoC training (RADicA dataset)
python "Model + HTML Asthma/src/train_model.py"

# TB demonstration (synthetic dataset)
python "Model + HTML TB Synthetic/train_model_tb.py"

# Classifier benchmark
python "Model + HTML Asthma/src/benchmark.py"

# Per-phase inference timing
python "Model + HTML Asthma/src/inference_timing.py"
```

---

## Headline results

### Asthma PoC (RADicA, 196 samples, 111 patients, real clinical data)

| Model | AUC (5-fold patient-grouped CV) | Balanced Accuracy |
|---|---|---|
| **SVM (RBF)** | 0.685 ± 0.10 | 0.623 |
| **Logistic Regression (L2)** | 0.678 ± 0.05 | 0.628 |
| **XGBoost (production)** | 0.678 ± 0.06 | 0.587 |
| **Random Forest (baseline)** | 0.667 ± 0.04 | 0.582 |
| Logistic Regression (L1) | 0.674 ± 0.06 | 0.608 |
| Quadratic Discriminant | 0.633 ± 0.06 | 0.604 |
| k-Nearest Neighbours | 0.642 ± 0.05 | 0.641 |
| Gaussian Naive Bayes | 0.618 ± 0.04 | 0.563 |

All models cluster within AUC 0.62–0.69 → the **headline AUC is realistic**, not a model-choice artifact. We chose XGBoost over the slightly-higher SVM because XGBoost has **lower variance** (0.06 vs 0.10), supports **native SHAP explanations**, and is **27× faster at inference** (1.2 ms vs 32 ms).

### TB demonstration (synthetic dataset, 300 samples, 8 VOCs × 4 features)

| Metric | RandomForest | XGBoost |
|---|---|---|
| 5-fold CV AUC | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Leave-one-batch-out AUC | 1.000 ± 0.000 | 1.000 ± 0.000 |

**Important:** AUC = 1.0 reflects clean synthetic-data separability, NOT clinical performance. This demonstrates the pipeline classifies correctly when given clean signal. Real-world TB validation pending data collection.

### Edge-device feasibility (inference timing)

| Phase | Median latency (ms) |
|---|---|
| Preprocessing (ALS + SavGol + peak detection) | 136 |
| StandardScaler + RFE transform | 0.3 |
| XGBoost prediction | 0.9 |
| Random Forest prediction | 34.6 |
| **End-to-end (XGBoost)** | **~138 ms** |

On a Raspberry Pi 4, expect 2–4× these times → end-to-end still **under 1 second per breath sample**. Deployable on inexpensive edge hardware.

---

## Datasets

This repo does **not** redistribute the source datasets. See `Model + HTML TB Synthetic/DataSources.md` for download links.

- **RADicA breath VOC dataset** (asthma PoC) — Figshare DOI `10.6084/m9.figshare.29504333`. Open-access. Patient-level breath-VOC peak tables, two analytical batches.
- **Synthetic microGC TB dataset** (TB demo) — generated to match microGC + PID hardware output. 300 samples, 8 literature-supported TB VOC biomarkers (Methyl Nicotinate, Isoprene, Pentane, Acetone, 2-Butanone, Hexane, Ethanol, Benzene). 4 features per VOC (retention time, peak area, peak height, signal-to-noise ratio).

---

## License

This codebase: MIT License (see `LICENSE`).
Datasets retain their own licenses — see `Model + HTML TB Synthetic/DataSources.md`.

---

## Citation

If this codebase is useful in your research, please cite:

```
Binus University TBreath Team (2026). TBreath: AI-assisted breathomics
for early tuberculosis screening. ASEAN AI Hackathon 2026,
Public Health & Telemedicine track, Binus University, Indonesia.
```

---

## Acknowledgments

Developed for the Public Health & Telemedicine track of the ASEAN AI Hackathon 2026. Code generation and pipeline architecture assistance from Anthropic's Claude (see `TBreath_AI_Use_Report.md` for full AI-use disclosure).
