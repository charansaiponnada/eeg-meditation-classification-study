import numpy as np
import pandas as pd
import mne
from  pathlib import Path

from mne.time_frequency import psd_array_welch
from scipy.stats import entropy
from scipy.signal import butter, filtfilt, hilbert, coherence
from sklearn.feature_selection import mutual_info_regression
import random

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


def band_power(eeg, sfreq):
    psd, freqs = psd_array_welch(eeg, sfreq, fmin=1, fmax=40, n_fft=1024)
    bands = {"delta":(1,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30)}
    return {
        f"bp_{b}": psd[:, (freqs>=f1)&(freqs<=f2)].mean()
        for b,(f1,f2) in bands.items()
    }

def entropy_feature(eeg):
    return {"entropy": np.mean([
        entropy(np.histogram(ch, bins=100, density=True)[0] + 1e-12)
        for ch in eeg
    ])}

def fast_mi(eeg, n_pairs=50):
    vals = []
    for _ in range(n_pairs):
        i,j = random.sample(range(eeg.shape[0]),2)
        vals.append(mutual_info_regression(
            eeg[i].reshape(-1,1), eeg[j])[0])
    return {"mi": np.mean(vals)}

def fast_coherence(eeg, sfreq, n_pairs=50):
    vals = []
    for _ in range(n_pairs):
        i,j = random.sample(range(eeg.shape[0]),2)
        f,C = coherence(eeg[i], eeg[j], fs=sfreq)
        vals.append(C[(f>=8)&(f<=13)].mean())
    return {"coh_alpha": np.mean(vals)}

def pac_feature(eeg, sfreq):
    def bp(x,f1,f2):
        b,a = butter(4,[f1/(sfreq/2),f2/(sfreq/2)],btype="band")
        return filtfilt(b,a,x)
    vals=[]
    for ch in eeg:
        phase = np.angle(hilbert(bp(ch,4,8)))
        amp   = np.abs(hilbert(bp(ch,13,30)))
        vals.append(np.corrcoef(np.sin(phase),amp)[0,1])
    return {"pac": np.nanmean(vals)}

def build_feature_dataset(bids_root):
    rows = []
    subject_id = 0

    for sub in sorted(bids_root.glob("sub-*")):
        subject_id += 1
        for ses in sorted(sub.glob("ses-*")):
            try:
                eeg_dir = ses / "eeg"
                eeg_file = next(eeg_dir.glob("*_eeg.bdf"))
                label_file = next(eeg_dir.glob("*_labels.tsv"))

                print(f"Processing {sub.name} {ses.name}")

                raw = preprocess_eeg(
                    eeg_file,
                    bids_root / "task-meditation_channels.tsv"
                )
                labels = pd.read_csv(label_file, sep="\t")
                sfreq = raw.info["sfreq"]

                for _,r in labels.iterrows():
                    if pd.isna(r["label"]): continue
                    start,end = r["onset"]-30.5, r["onset"]-0.5
                    if start<0: continue

                    eeg = raw.copy().crop(start,end).get_data()

                    feats={}
                    feats.update(band_power(eeg,sfreq))
                    feats.update(entropy_feature(eeg))
                    feats.update(fast_mi(eeg))
                    feats.update(fast_coherence(eeg,sfreq))
                    feats.update(pac_feature(eeg,sfreq))

                    feats["label"] = int(r["label"])
                    feats["subject"] = subject_id
                    rows.append(feats)

                print(f"  → {len(rows)} samples so far")
                print(f"Done: {sub.name} {ses.name} has {len(rows)} samples")
            except StopIteration:
                print(f"Skipping {sub.name} {ses.name}")

    return pd.DataFrame(rows)

def main():
    bids_root = Path("C:/projects/eegnets/ds001787")
    df = build_feature_dataset(bids_root)
    df.to_csv("features_all_subjects.csv", index=False)
    print("Saved features_all_subjects.csv")

if __name__ == "__main__":
    main()
