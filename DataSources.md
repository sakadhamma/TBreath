# Datasets

This folder does not contain the raw breath-VOC datasets. The training scripts in `../src/` load them from `/mnt/user-data/uploads/` (the original development path) — adjust the `UP = Path(...)` constant at the top of each script to point to wherever you've placed the downloaded files.

## 1. RADicA breath-VOC dataset (asthma PoC)

- **Source:** Figshare
- **DOI:** [10.6084/m9.figshare.29504333](https://doi.org/10.6084/m9.figshare.29504333)
- **License:** Open access (terms on Figshare landing page)
- **Files used by `src/train_model_asthma.py`:**
  - `RADicA_BG_adjusted_B1_outl_removed.csv` — Batch 1, background-corrected, outliers removed (91 samples, 142 VOCs)
  - `RADicA_BG_adjusted_B2_outl_removed.csv` — Batch 2, same processing (105 samples, 142 VOCs)
- **What it contains:** patient-level breath VOC peak intensities (one row per sample), with diagnosis labels (Asthma / Not Asthma), patient ID, and core-visit identifier.
- **What it does NOT contain:** clinical PII has already been removed by the dataset authors.

## 2. Synthetic microGC TB dataset (TB demonstration)

- **Format:** Excel (.xlsx) — provided to the team by a collaborator
- **What it contains:** 300 simulated samples (150 TB-positive, 150 controls) across 5 synthetic analytical batches. Each sample has Retention Time, Peak Area, Peak Height, and Signal-to-Noise Ratio for 8 literature-supported TB VOC biomarkers (Methyl Nicotinate, Isoprene, Pentane, Acetone, 2-Butanone, Hexane, Ethanol, Benzene).
- **Important:** this dataset is **synthetic**, not real patients. High classification accuracy on this data reflects the cleanly-separable nature of synthetic patterns, not clinical performance.
- **File used by `src/train_model_tb.py`:** `synthetic_microgc_rawstyle_tb_dataset.xlsx`

## 3. Future: Real TB patient data

TB patient breath-VOC collection is planned. The training pipeline transfers directly once collected: it expects the same per-VOC peak feature format already used by the synthetic dataset.
