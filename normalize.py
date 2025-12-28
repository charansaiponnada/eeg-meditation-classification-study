import pandas as pd
import numpy as np

df = pd.read_csv("features_all_subjects.csv")

feature_cols = [c for c in df.columns if c not in ["label", "subject"]]

df_norm = df.copy()

for subj in df["subject"].unique():
    idx = df["subject"] == subj
    df_norm.loc[idx, feature_cols] = (
        df.loc[idx, feature_cols] - df.loc[idx, feature_cols].mean()
    ) / (df.loc[idx, feature_cols].std() + 1e-8)

print("Subject-wise normalization done.")
