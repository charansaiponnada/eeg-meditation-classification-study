# ===============================
"""
Fix bandpower to be region-wise:

Example:
1. define the regions (Frontal, Central, Parietal, Occipital)
2. ⁠compute mean log bandpower per region
3. ⁠compute relative power per region (band / total 1–40)
4. ⁠add ratios: theta/beta, alpha/theta per region

Use 10 seconds windows only instead of the current 30 seconds
"""
# ===============================

import numpy as np
import pandas as pd
from pathlib import Path
import mne
from mne.time_frequency import psd_array_welch

# -------------------------------
# CONFIG
# -------------------------------

BIDS_ROOT = Path("C:/projects/eegnets/ds001787")

SFREQ = 256
WINDOW_SEC = 10.0
OVERLAP = 0.5

WINDOW_TMIN = -15.0
WINDOW_TMAX = -5.0

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30)
}

REGIONS = {
    "Frontal":   ["Fp1","Fp2","F3","F4","F7","F8","Fz"],
    "Central":   ["C3","C4","Cz"],
    "Parietal":  ["P3","P4","Pz"],
    "Occipital": ["O1","O2"]
}

# -------------------------------
# PREPROCESSING
# -------------------------------

def preprocess_eeg(eeg_path: Path, channels_tsv: Path):
    raw = mne.io.read_raw_bdf(eeg_path, preload=True, verbose=False)

    # Load channel names
    df_ch = pd.read_csv(channels_tsv, sep="\t")
    eeg_names = df_ch[df_ch["type"] == "EEG"]["name"].tolist()
    assert len(eeg_names) == 64

    # Rename & pick EEG channels
    rename_map = dict(zip(raw.ch_names[:64], eeg_names))
    raw.rename_channels(rename_map)
    raw.pick_channels(eeg_names)

    # Montage & preprocessing
    raw.set_montage("standard_1020", match_case=False)
    raw.resample(SFREQ)
    raw.filter(1, 40)
    raw.notch_filter(50)
    raw.set_eeg_reference("average")

    return raw

# -------------------------------
# WINDOW EXTRACTION
# -------------------------------

def extract_windows(raw, labels_df):
    step = WINDOW_SEC * (1 - OVERLAP)
    X, y = [], []

    for _, row in labels_df.iterrows():
        if pd.isna(row["label"]):
            continue

        t0 = row["onset"] + WINDOW_TMIN
        t1 = row["onset"] + WINDOW_TMAX


        t = t0
        while t + WINDOW_SEC <= t1:
            seg = raw.copy().crop(t, t + WINDOW_SEC)
            if seg.n_times > 0:
                X.append(seg.get_data())
                y.append(int(row["label"]))
            t += step

    return np.array(X), np.array(y)

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------

def region_features(window_data, sfreq, ch_names):
    feats = {}

    for region, channels in REGIONS.items():
        picks = [ch_names.index(c) for c in channels if c in ch_names]
        data = window_data[picks]

        psd, freqs = psd_array_welch(data, sfreq, fmin=1, fmax=40, n_fft=1024)
        psd = psd.mean(axis=0)

        total_power = psd.sum()

        band_powers = {}
        for band, (fmin, fmax) in BANDS.items():
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            bp = psd[idx].sum()
            band_powers[band] = bp

            feats[f"{region}_logbp_{band}"] = np.log(bp + 1e-10)
            feats[f"{region}_relbp_{band}"] = bp / (total_power + 1e-10)

        feats[f"{region}_theta_beta"] = band_powers["theta"] / (band_powers["beta"] + 1e-10)
        feats[f"{region}_alpha_theta"] = band_powers["alpha"] / (band_powers["theta"] + 1e-10)

    return feats

# -------------------------------
# MAIN LOOP
# -------------------------------

def main():
    rows = []

    for sub_dir in sorted(BIDS_ROOT.glob("sub-*")):
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            eeg_dir = ses_dir / "eeg"
            try:
                eeg_file = next(eeg_dir.glob("*_eeg.bdf"))
                label_file = next(eeg_dir.glob("*_labels.tsv"))
                ch_tsv = BIDS_ROOT / "task-meditation_channels.tsv"

                raw = preprocess_eeg(eeg_file, ch_tsv)

                labels_df = pd.read_csv(label_file, sep="\t")
                X, y = extract_windows(raw, labels_df)

                for i in range(len(X)):
                    feats = region_features(X[i], raw.info["sfreq"], raw.ch_names)
                    feats["label"] = y[i]
                    feats["subject"] = int(sub_dir.name.split("-")[1])
                    rows.append(feats)

            except StopIteration:
                continue

    df = pd.DataFrame(rows)
    df.to_csv("features_region_all_subjects.csv", index=False)
    print("Saved features_region_all_subjects.csv")

# -------------------------------
if __name__ == "__main__":
    main()
