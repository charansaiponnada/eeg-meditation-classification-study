# =====================================================
"""
For Riemannian:

1. Compute covariance per trial window (64×64)
2. ⁠map to tangent space
3. ⁠train LogisticRegression / SVM
"""
# =====================================================

import numpy as np
import pandas as pd
from pathlib import Path
import mne

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -------------------------
# CONFIG
# -------------------------

BIDS_ROOT = Path("C:/projects/eegnets/ds001787")

SFREQ = 256
WINDOW_SEC = 10.0
OVERLAP = 0.5

WINDOW_TMIN = -15.0
WINDOW_TMAX = -5.0


# -------------------------
# PREPROCESSING
# -------------------------

def preprocess_eeg(eeg_path, channels_tsv):
    raw = mne.io.read_raw_bdf(eeg_path, preload=True, verbose=False)

    df_ch = pd.read_csv(channels_tsv, sep="\t")
    eeg_names = df_ch[df_ch["type"] == "EEG"]["name"].tolist()
    assert len(eeg_names) == 64

    rename_map = dict(zip(raw.ch_names[:64], eeg_names))
    raw.rename_channels(rename_map)
    raw.pick_channels(eeg_names)

    raw.set_montage("standard_1020", match_case=False)
    raw.resample(SFREQ)
    raw.filter(1, 40)
    raw.notch_filter(50)
    raw.set_eeg_reference("average")

    return raw


# -------------------------
# WINDOW EXTRACTION
# -------------------------

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


# -------------------------
# LOAD DATASET
# -------------------------

def load_dataset():
    X_all, y_all, groups = [], [], []

    for sub_dir in sorted(BIDS_ROOT.glob("sub-*")):
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            eeg_dir = ses_dir / "eeg"
            try:
                eeg_file = next(eeg_dir.glob("*_eeg.bdf"))
                label_file = next(eeg_dir.glob("*_labels.tsv"))
                ch_tsv = BIDS_ROOT / "task-meditation_channels.tsv"

                print(f"Loading {sub_dir.name} {ses_dir.name}")

                raw = preprocess_eeg(eeg_file, ch_tsv)
                labels_df = pd.read_csv(label_file, sep="\t")

                X, y = extract_windows(raw, labels_df)

                X_all.append(X)
                y_all.append(y)
                groups.extend([int(sub_dir.name.split("-")[1])] * len(y))

            except StopIteration:
                continue

    return np.vstack(X_all), np.hstack(y_all), np.array(groups)


# -------------------------
# MODEL TRAINING
# -------------------------

def train_model(name, clf, X, y, groups):
    print(f"\n=== {name} ===")

    gkf = GroupKFold(n_splits=5)
    scores = []

    pipe = Pipeline([
        ("cov", Covariances(estimator="lwf")),
        ("ts", TangentSpace(metric="riemann")),
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), start=1):
        pipe.fit(X[tr], y[tr])
        yp = pipe.predict(X[te])

        acc  = accuracy_score(y[te], yp)
        prec = precision_score(y[te], yp)
        rec  = recall_score(y[te], yp)
        f1   = f1_score(y[te], yp)

        scores.append([acc, prec, rec, f1])

        print(
            f"[Fold {fold}] "
            f"ACC={acc:.3f} | "
            f"P={prec:.3f} | "
            f"R={rec:.3f} | "
            f"F1={f1:.3f}"
        )

    scores = np.array(scores)
    mean = scores.mean(axis=0)
    std  = scores.std(axis=0)

    print(
        f"{name} SUMMARY:\n"
        f"ACC : {mean[0]:.4f} ± {std[0]:.4f}\n"
        f"PREC: {mean[1]:.4f} ± {std[1]:.4f}\n"
        f"REC : {mean[2]:.4f} ± {std[2]:.4f}\n"
        f"F1  : {mean[3]:.4f} ± {std[3]:.4f}"
    )

    return mean


# -------------------------
# MAIN
# -------------------------

def main():
    X, y, groups = load_dataset()

    print("\n=== Riemannian feature extraction ===")

    results = []

    results.append(
        ["LogisticRegression", *train_model(
            "LogisticRegression",
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            X, y, groups
        )]
    )

    results.append(
        ["SVM", *train_model(
            "SVM",
            SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced"),
            X, y, groups
        )]
    )

    df = pd.DataFrame(results, columns=["Model", "ACC", "PREC", "REC", "F1"])

    print("\n=== FINAL COMPARISON (RIEMANNIAN) ===")
    print(df.sort_values("F1", ascending=False))


if __name__ == "__main__":
    main()
