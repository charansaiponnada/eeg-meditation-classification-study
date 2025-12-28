import pandas as pd
import numpy as np


df_norm = pd.read_csv("features_all_subjects.csv")
df_feat = df_norm.copy()

# Theta / Alpha ratio
df_feat["theta_alpha_ratio"] = (
    df_feat["bp_theta"] / (df_feat["bp_alpha"] + 1e-12)
)

# Relative band powers
total_power = (
    df_feat["bp_delta"] +
    df_feat["bp_theta"] +
    df_feat["bp_alpha"] +
    df_feat["bp_beta"] + 1e-12
)

df_feat["rel_delta"] = df_feat["bp_delta"] / total_power
df_feat["rel_theta"] = df_feat["bp_theta"] / total_power
df_feat["rel_alpha"] = df_feat["bp_alpha"] / total_power
df_feat["rel_beta"]  = df_feat["bp_beta"]  / total_power

print("Ratio and relative power features added.")
