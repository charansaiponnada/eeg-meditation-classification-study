import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


df = pd.read_csv("features_all_subjects.csv")

X = df.drop(columns=["label", "subject"]).values
y = df["label"].values
groups = df["subject"].values


models = {
    "LogisticRegression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear"
    ),

    "SVM": SVC(
        kernel="rbf",
        C=2.0,
        gamma="scale",
        class_weight="balanced",
        probability=False
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        random_state=42
    ),

    "CatBoost": CatBoostClassifier(
        iterations=400,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        verbose=0,
        random_seed=42
    )
}



gkf = GroupKFold(n_splits=5)

results = []

for model_name, model in models.items():
    print(f"\n=== {model_name} ===")

    fold = 0
    metrics = []

    for train_idx, test_idx in gkf.split(X, y, groups):
        fold += 1

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", model)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        metrics.append([acc, prec, rec, f1])

        print(
            f"[Fold {fold}] "
            f"ACC={acc:.3f} | "
            f"P={prec:.3f} | "
            f"R={rec:.3f} | "
            f"F1={f1:.3f}"
        )

    metrics = np.array(metrics)
    mean = metrics.mean(axis=0)
    std  = metrics.std(axis=0)

    print(
        f"\n{model_name} SUMMARY:\n"
        f"ACC : {mean[0]:.4f} ± {std[0]:.4f}\n"
        f"PREC: {mean[1]:.4f} ± {std[1]:.4f}\n"
        f"REC : {mean[2]:.4f} ± {std[2]:.4f}\n"
        f"F1  : {mean[3]:.4f} ± {std[3]:.4f}"
    )

    results.append([model_name, *mean])


results_df = pd.DataFrame(
    results,
    columns=["Model", "ACC", "PREC", "REC", "F1"]
)

print("\n=== FINAL COMPARISON ===")
print(results_df.sort_values("F1", ascending=False))
