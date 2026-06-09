"""
Preprocessing Demo — synthetic GC-PID chromatogram.

The RADicA cleaned files (B1/B2) are already post-peak-detection peak tables
(one number per VOC per sample). The actual chromatogram traces aren't shared.
This script generates a REPRESENTATIVE chromatogram with realistic structure
(noisy baseline drift + Gaussian VOC peaks), then demonstrates each stage of
the proposed on-device preprocessing pipeline:

    1. ALS  (Asymmetric Least Squares) baseline correction
    2. Savitzky-Golay smoothing
    3. Prominence-based peak detection

Output: /home/claude/artifacts/preprocessing_demo.png  (used by the dashboard)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.sparse import csc_matrix, diags, eye
from scipy.sparse.linalg import spsolve

OUT = Path("/home/claude/artifacts"); OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(7)

# ----- Simulate a 50-minute chromatogram at 5 Hz (15000 points) -----
T = np.linspace(0, 50, 15000)           # minutes
true_baseline = (0.6 + 0.4 * np.exp(T / 30) +
                 0.15 * np.sin(2 * np.pi * T / 18))   # slow drift + curvature

# Plant our 12 selected biomarkers at plausible retention times.
biomarker_times = {
    "Pentane":             3.8,
    "Trichlorofluoromethane": 5.1,
    "Azetidine":           7.4,
    "3-methylpentane":     9.0,
    "Ethyl butanoate":    11.6,
    "Hexane,2,2,5-trimethyl-": 13.2,
    "Cis-1-Ethyl-3-methyl-cyclohexane": 17.8,
    "Pyridine":           20.4,
    "Benzonitrile":       24.0,
    "Styrene":            27.5,
    "Propylbenzene":      30.9,
    "Unknown C12H24":     35.6,
}
heights = rng.uniform(2.0, 5.5, size=len(biomarker_times))

signal = np.copy(true_baseline)
for (name, rt), h in zip(biomarker_times.items(), heights):
    sigma = 0.05 + 0.01 * rng.random()
    signal += h * np.exp(-((T - rt) ** 2) / (2 * sigma ** 2))

# Add a few "junk" peaks (column bleed / solvent artifacts) the system should ignore.
for rt in [1.2, 2.1, 47.5, 48.8]:
    signal += rng.uniform(0.3, 0.8) * np.exp(-((T - rt) ** 2) / (2 * 0.06 ** 2))

# Add detector noise.
noise = rng.normal(0, 0.04, size=T.shape) + rng.normal(0, 0.10, size=T.shape) * (rng.random(T.shape) < 0.005)
raw = signal + noise


# ---------------- Stage 1: ALS baseline correction --------------------------
def als_baseline(y, lam=1e6, p=0.005, niter=10):
    """Asymmetric least squares (Eilers & Boelens 2005).
       Large lam => smoother baseline. Small p => baseline hugs lower envelope."""
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    DTD = lam * (D @ D.T)
    w = np.ones(L)
    for _ in range(niter):
        W = diags(w, 0, shape=(L, L))
        Z = csc_matrix(W + DTD)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

baseline = als_baseline(raw, lam=1e7, p=0.001, niter=10)
corrected = raw - baseline


# ---------------- Stage 2: Savitzky-Golay smoothing -------------------------
smoothed = savgol_filter(corrected, window_length=51, polyorder=3)


# ---------------- Stage 3: prominence-based peak detection ------------------
peaks, props = find_peaks(smoothed, prominence=0.3, distance=200)
peak_times = T[peaks]
peak_heights = smoothed[peaks]


# ---------------- Figure -----------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True,
                         gridspec_kw={"hspace": 0.18})

panel = dict(color="#2c2c2a", linewidth=0.9)
accent = "#3266ad"
ok_green = "#1D9E75"

axes[0].plot(T, raw, **panel)
axes[0].plot(T, baseline, color=accent, linewidth=1.4, label="ALS baseline")
axes[0].set_ylabel("PID signal", fontsize=10)
axes[0].set_title("1. Raw chromatogram + ALS baseline estimate", loc="left",
                  fontsize=11, color="#2c2c2a")
axes[0].legend(loc="upper left", frameon=False, fontsize=9)
axes[0].grid(alpha=0.15)

axes[1].plot(T, corrected, color="#666661", linewidth=0.8, alpha=0.6, label="Baseline-subtracted")
axes[1].plot(T, smoothed, color="#2c2c2a", linewidth=1.0, label="Savitzky-Golay smoothed")
axes[1].set_ylabel("PID signal", fontsize=10)
axes[1].set_title("2. Baseline-subtracted + Savitzky-Golay smoothing", loc="left",
                  fontsize=11, color="#2c2c2a")
axes[1].legend(loc="upper left", frameon=False, fontsize=9)
axes[1].grid(alpha=0.15)

axes[2].plot(T, smoothed, **panel)
axes[2].plot(peak_times, peak_heights, "o", color=ok_green, markersize=6,
             markeredgecolor="white", markeredgewidth=1.2,
             label=f"Detected peaks (n={len(peaks)})")
# Annotate each detected peak with the closest biomarker label.
bm_names = list(biomarker_times.keys())
bm_rts = np.array(list(biomarker_times.values()))
for px, py in zip(peak_times, peak_heights):
    idx = np.argmin(np.abs(bm_rts - px))
    if abs(bm_rts[idx] - px) < 0.5:
        axes[2].annotate(bm_names[idx], (px, py), xytext=(0, 8),
                         textcoords="offset points", fontsize=7.5,
                         ha="center", color="#2c2c2a", rotation=0)
axes[2].set_ylabel("PID signal", fontsize=10)
axes[2].set_xlabel("Retention time (minutes)", fontsize=10)
axes[2].set_title("3. Prominence-based peak detection (extracted biomarker features)",
                  loc="left", fontsize=11, color="#2c2c2a")
axes[2].legend(loc="upper left", frameon=False, fontsize=9)
axes[2].grid(alpha=0.15)

for ax in axes:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

plt.suptitle("On-device preprocessing pipeline (representative GC-PID signal)",
             fontsize=13, color="#2c2c2a", x=0.125, y=0.985, ha="left",
             fontweight="normal")
plt.savefig(OUT / "preprocessing_demo.png", dpi=140, bbox_inches="tight",
            facecolor="white")
plt.close()

print(f"  Detected {len(peaks)} peaks at retention times: "
      f"{np.round(peak_times, 2).tolist()}")
print(f"  Saved {OUT/'preprocessing_demo.png'}")
