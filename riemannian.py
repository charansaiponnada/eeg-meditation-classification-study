# ===============================
# riemannian.py
# ===============================

import numpy as np
import pandas as pd
from pathlib import Path

import mne
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

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
# USER PATHS (adjust if needed)
# -------------------------------------------------
BIDS_ROOT = Path("C:/projects/eegnets/ds001787")
WINDOW_TMIN = -30.5 
WINDOW_TMAX = -0.5
SFREQ = 256


# -------------------------------------------------
# EEG preprocessing (already validated by you)
# -------------------------------------------------
def preprocess_eeg(eeg_path, channels_tsv):
    import pandas as pd
    import mne

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
# Extract probe-locked EEG windows
# -------------------------------------------------
def extract_windows(raw, labels_df):
    X, y, groups = [], [], []

    for _, row in labels_df.iterrows():

        # use only labeled stimulus rows
        if row["trial_type"] != "stimulus":
            continue
        if pd.isna(row["label"]):
            continue

        probe_time = row["onset"]

        start = probe_time + WINDOW_TMIN
        end   = probe_time + WINDOW_TMAX

        if start < 0:
            continue

        data = raw.copy().crop(start, end).get_data()

        X.append(data)
        y.append(int(row["label"]))
        groups.append(int(row["subject"]))

    return np.array(X), np.array(y), np.array(groups)



# -------------------------------------------------
# Load all subjects and sessions
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

                X, y, g = extract_windows(raw, labels_df)

                X_all.append(X)
                y_all.append(y)
                g_all.append(g)

            except StopIteration:
                continue

    return (
        np.concatenate(X_all),
        np.concatenate(y_all),
        np.concatenate(g_all),
    )


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    print("\n=== Loading EEG windows ===")
    X, y, groups = load_dataset()

    print("EEG windows:", X.shape)
    print("Labels:", np.unique(y, return_counts=True))

    # Riemannian feature extraction
    print("\n=== Riemannian feature extraction ===")
    cov = Covariances(estimator="oas").fit_transform(X)
    ts = TangentSpace(metric="riemann")
    X_feat = ts.fit_transform(cov)

    # Models
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="liblinear"
        ),
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
            random_seed=42,
        ),
    }

    gkf = GroupKFold(n_splits=5)
    results = []

    for name, model in models.items():
        print(f"\n=== {name} ===")
        scores = []

        for fold, (tr, te) in enumerate(gkf.split(X_feat, y, groups), 1):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", model),
            ])

            pipe.fit(X_feat[tr], y[tr])
            y_pred = pipe.predict(X_feat[te])

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
    print("\n=== FINAL COMPARISON (RIEMANNIAN) ===")
    print(df_res.sort_values("F1", ascending=False))


if __name__ == "__main__":
    main()
