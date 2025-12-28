# ===============================
# region_based.py
# ===============================

import numpy as np
import pandas as pd
from pathlib import Path

import mne
from mne.time_frequency import psd_array_welch

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -------------------------------------------------
# PATHS & PARAMS
# -------------------------------------------------
BIDS_ROOT = Path("C:/projects/eegnets/ds001787")
WINDOW_TMIN = -30.5
WINDOW_TMAX = -0.5
SFREQ = 256


# -------------------------------------------------
# Channel regions (10–20 system)
# -------------------------------------------------
REGIONS = {
    "frontal": ["Fp1","Fp2","AF3","AF4","F1","F2","F3","F4","Fz"],
    "central": ["FC1","FC2","C1","C2","C3","C4","Cz"],
    "parietal": ["CP1","CP2","P1","P2","P3","P4","Pz","PO3","PO4","POz"],
    "occipital": ["O1","O2","Oz"],
}

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
}


# -------------------------------------------------
# EEG preprocessing
# -------------------------------------------------
def preprocess_eeg(eeg_path, channels_tsv):
    # Load raw EEG
    raw = mne.io.read_raw_bdf(eeg_path, preload=True, verbose=False)

    # Load BIDS channels.tsv
    df_ch = pd.read_csv(channels_tsv, sep="\t")
    eeg_names = df_ch[df_ch["type"] == "EEG"]["name"].tolist()

    if len(eeg_names) != 64:
        raise RuntimeError(f"Expected 64 EEG channels, got {len(eeg_names)}")

    # --- CRITICAL PART ---
    # Identify BioSemi EEG channels (A1–B32, B1–B32)
    biosig_eeg = raw.ch_names[:64]   # BioSemi guarantees EEG first

    # Rename BioSemi channels → real EEG names
    rename_map = dict(zip(biosig_eeg, eeg_names))
    raw.rename_channels(rename_map)

    # Sanity check BEFORE picking
    missing = set(eeg_names) - set(raw.ch_names)
    if missing:
        raise RuntimeError(f"Missing EEG channels after renaming: {missing}")

    # Now safely pick EEG channels
    raw.pick_channels(eeg_names)

    # Set montage
    raw.set_montage("standard_1020", match_case=False)

    # Resample (safe if already 256 Hz)
    raw.resample(256)

    # Filtering
    raw.filter(1, 40)
    raw.notch_filter(50)

    # Average reference
    raw.set_eeg_reference("average")

    return raw

# -------------------------------------------------
# Extract region-based features
# -------------------------------------------------
def extract_region_features(eeg, ch_names):
    psd, freqs = psd_array_welch(eeg, SFREQ, fmin=1, fmax=40, n_fft=1024)

    feats = {}

    for band, (f1, f2) in BANDS.items():
        f_idx = (freqs >= f1) & (freqs <= f2)

        for region, channels in REGIONS.items():
            ch_idx = [ch_names.index(c) for c in channels if c in ch_names]
            if not ch_idx:
                continue

            feats[f"{band}_{region}"] = psd[ch_idx][:, f_idx].mean()

    return feats


# -------------------------------------------------
# Load windows & features
# -------------------------------------------------
def load_dataset():
    X_all, y_all, g_all = [], [], []

    for sub_dir in sorted(BIDS_ROOT.glob("sub-*")):
        subj_id = int(sub_dir.name.split("-")[1])

        for ses_dir in sorted(sub_dir.glob("ses-*")):
            eeg_dir = ses_dir / "eeg"
            try:
                eeg_file = next(eeg_dir.glob("*_task-meditation_eeg.bdf"))
                labels_file = next(eeg_dir.glob("*_labels.tsv"))
                channels_tsv = BIDS_ROOT / "task-meditation_channels.tsv"

                print(f"Loading {sub_dir.name} {ses_dir.name}")

                raw = preprocess_eeg(eeg_file, channels_tsv)
                labels_df = pd.read_csv(labels_file, sep="\t")
                labels_df["subject"] = subj_id

                for _, row in labels_df.iterrows():
                    if row["trial_type"] != "stimulus" or pd.isna(row["label"]):
                        continue

                    t0 = row["onset"] + WINDOW_TMIN
                    t1 = row["onset"] + WINDOW_TMAX
                    if t0 < 0:
                        continue

                    eeg = raw.copy().crop(t0, t1).get_data()
                    feats = extract_region_features(eeg, raw.ch_names)

                    X_all.append(list(feats.values()))
                    y_all.append(int(row["label"]))
                    g_all.append(subj_id)

            except StopIteration:
                continue

    feature_names = list(feats.keys())
    return np.array(X_all), np.array(y_all), np.array(g_all), feature_names


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    X, y, groups, feature_names = load_dataset()

    print("\nSamples:", X.shape)
    print("Labels:", np.unique(y, return_counts=True))

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "SVM": SVC(kernel="rbf", C=2.0, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
            random_state=42,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            verbose=0,
        ),
    }

    gkf = GroupKFold(n_splits=5)
    results = []

    for name, model in models.items():
        print(f"\n=== {name} ===")
        scores = []

        for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", model),
            ])

            pipe.fit(X[tr], y[tr])
            y_pred = pipe.predict(X[te])

            acc = accuracy_score(y[te], y_pred)
            prec = precision_score(y[te], y_pred)
            rec = recall_score(y[te], y_pred)
            f1 = f1_score(y[te], y_pred)

            scores.append([acc, prec, rec, f1])
            print(f"[Fold {fold}] ACC={acc:.3f} | P={prec:.3f} | R={rec:.3f} | F1={f1:.3f}")

        scores = np.array(scores)
        mean = scores.mean(axis=0)
        std = scores.std(axis=0)

        print(
            f"{name} SUMMARY:\n"
            f"ACC : {mean[0]:.4f} ± {std[0]:.4f}\n"
            f"PREC: {mean[1]:.4f} ± {std[1]:.4f}\n"
            f"REC : {mean[2]:.4f} ± {std[2]:.4f}\n"
            f"F1  : {mean[3]:.4f} ± {std[3]:.4f}"
        )

        results.append([name, *mean])

    df_res = pd.DataFrame(results, columns=["Model", "ACC", "PREC", "REC", "F1"])
    print("\n=== FINAL COMPARISON (REGION-BASED) ===")
    print(df_res.sort_values("F1", ascending=False))


if __name__ == "__main__":
    main()
