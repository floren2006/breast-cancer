"""
Script untuk menyiapkan seluruh artefak (dataset, model, statistik benchmark,
hasil tuning) yang dipakai oleh aplikasi Streamlit.
Dijalankan SEKALI saat development -- hasilnya disimpan ke folder model/.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from scipy.stats import loguniform
import optuna
import warnings

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42

# ----------------------------------------------------------------------
# 1. Build data.csv in the same column format as the original notebook
#    (radius_mean, texture_mean, ..., concave points_mean, ..., id, diagnosis)
# ----------------------------------------------------------------------
raw = load_breast_cancer()
feat_names = raw.feature_names  # e.g. 'mean radius', 'radius error', 'worst radius'


def rename(col):
    if col.startswith("mean "):
        base = col[5:]
        suffix = "mean"
    elif col.endswith(" error"):
        base = col[: -len(" error")]
        suffix = "se"
    elif col.startswith("worst "):
        base = col[6:]
        suffix = "worst"
    else:
        base, suffix = col, ""
    base = base.replace("concave points", "concave points")  # keep space, just clarity
    return f"{base}_{suffix}" if suffix else base


new_cols = [rename(c) for c in feat_names]
df = pd.DataFrame(raw.data, columns=new_cols)
df.insert(0, "id", np.arange(100000, 100000 + len(df)))
df["diagnosis"] = np.where(raw.target == 0, "M", "B")  # sklearn: 0=malignant,1=benign -> flip
# sklearn target_names = ['malignant' 'benign'] -> target 0 = malignant
df["Unnamed: 32"] = np.nan

# Reorder to match typical Kaggle layout: id, diagnosis, <30 features>, Unnamed: 32
ordered_cols = ["id", "diagnosis"] + new_cols + ["Unnamed: 32"]
df = df[ordered_cols]
df.to_csv("model/data.csv", index=False)
print("data.csv saved:", df.shape)

# ----------------------------------------------------------------------
# 2. Preprocessing -- identical strategy to the notebook (anti data leakage)
# ----------------------------------------------------------------------
df_raw = pd.read_csv("model/data.csv")
df_model = df_raw.copy()
df_model.drop(columns=["id", "Unnamed: 32"], inplace=True, errors="ignore")
df_model["diagnosis"] = df_model["diagnosis"].map({"M": 1, "B": 0})

X = df_model.drop(columns=["diagnosis"])
y = df_model["diagnosis"]
FEATURE_NAMES = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

X_train_df = X_train.copy()
X_train_df["diagnosis"] = y_train.values
mean_benign = X_train_df[X_train_df["diagnosis"] == 0].drop(columns="diagnosis").mean()
mean_malignant = X_train_df[X_train_df["diagnosis"] == 1].drop(columns="diagnosis").mean()
std_benign = X_train_df[X_train_df["diagnosis"] == 0].drop(columns="diagnosis").std()
std_malignant = X_train_df[X_train_df["diagnosis"] == 1].drop(columns="diagnosis").std()

X_test_df = X_test.copy()
X_test_df["true_label"] = y_test.values


def eval_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }


# ----------------------------------------------------------------------
# 3. Default model (baseline)
# ----------------------------------------------------------------------
pipe_default = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
])
pipe_default.fit(X_train, y_train)
y_def = pipe_default.predict(X_test)
y_def_prob = pipe_default.predict_proba(X_test)[:, 1]
baseline_metrics = eval_metrics(y_test, y_def, y_def_prob)

# ----------------------------------------------------------------------
# 4. GridSearchCV
# ----------------------------------------------------------------------
param_grid = [
    {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100], "clf__penalty": ["l2"], "clf__solver": ["lbfgs", "liblinear"]},
    {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100], "clf__penalty": ["l1"], "clf__solver": ["liblinear"]},
]
gs = GridSearchCV(
    Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))]),
    param_grid, cv=cv, scoring="roc_auc", n_jobs=-1,
)
gs.fit(X_train, y_train)
y_gs = gs.best_estimator_.predict(X_test)
y_gs_prob = gs.best_estimator_.predict_proba(X_test)[:, 1]
gs_metrics = eval_metrics(y_test, y_gs, y_gs_prob)

# ----------------------------------------------------------------------
# 5. RandomizedSearchCV
# ----------------------------------------------------------------------
rs = RandomizedSearchCV(
    Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))]),
    {"clf__C": loguniform(1e-3, 1e2), "clf__penalty": ["l2"], "clf__solver": ["lbfgs", "liblinear"]},
    n_iter=50, cv=cv, scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=-1,
)
rs.fit(X_train, y_train)
y_rs = rs.best_estimator_.predict(X_test)
y_rs_prob = rs.best_estimator_.predict_proba(X_test)[:, 1]
rs_metrics = eval_metrics(y_test, y_rs, y_rs_prob)

# ----------------------------------------------------------------------
# 6. Optuna
# ----------------------------------------------------------------------
def objective(trial):
    C = trial.suggest_float("C", 1e-4, 1e3, log=True)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    pipe_t = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, penalty=penalty, solver="liblinear", random_state=RANDOM_STATE, max_iter=1000)),
    ])
    return cross_val_score(pipe_t, X_train, y_train, cv=cv, scoring="roc_auc").mean()


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=100, show_progress_bar=False)
bp = study.best_params
bp["solver"] = "liblinear"
pipe_optuna = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=bp["C"], penalty=bp["penalty"], solver=bp["solver"], random_state=RANDOM_STATE, max_iter=1000)),
])
pipe_optuna.fit(X_train, y_train)
y_opt = pipe_optuna.predict(X_test)
y_opt_prob = pipe_optuna.predict_proba(X_test)[:, 1]
opt_metrics = eval_metrics(y_test, y_opt, y_opt_prob)

# ----------------------------------------------------------------------
# 7. Pilih model terbaik (Recall -> F1 -> ROC-AUC)
# ----------------------------------------------------------------------
all_results = pd.DataFrame([
    {"Metode": "Default (Baseline)", **baseline_metrics},
    {"Metode": "GridSearchCV", **gs_metrics},
    {"Metode": "RandomizedSearchCV", **rs_metrics},
    {"Metode": "Optuna (Bayesian)", **opt_metrics},
])

tuned_only = all_results.iloc[1:].copy()
best_idx = tuned_only.apply(
    lambda row: (row["Recall"], row["F1 Score"], row["ROC-AUC"], -list(tuned_only.index).index(row.name)),
    axis=1,
).idxmax()
best_row = all_results.loc[best_idx]
best_name = best_row["Metode"]

model_map = {
    "Default (Baseline)": (pipe_default, y_def, y_def_prob),
    "GridSearchCV": (gs.best_estimator_, y_gs, y_gs_prob),
    "RandomizedSearchCV": (rs.best_estimator_, y_rs, y_rs_prob),
    "Optuna (Bayesian)": (pipe_optuna, y_opt, y_opt_prob),
}
best_model, y_pred, y_pred_prob = model_map[best_name]
best_metrics = {
    "Accuracy": best_row["Accuracy"], "Precision": best_row["Precision"],
    "Recall": best_row["Recall"], "F1 Score": best_row["F1 Score"], "ROC-AUC": best_row["ROC-AUC"],
}

print("Model terbaik:", best_name)
print(best_metrics)

# ----------------------------------------------------------------------
# 8. Insight 1 -- coefficient / odds ratio
# ----------------------------------------------------------------------
coef = best_model.named_steps["clf"].coef_[0]
odds_ratio = np.exp(coef)
coef_df = pd.DataFrame({
    "Fitur": FEATURE_NAMES,
    "Koefisien": coef.round(4),
    "Odds Ratio": odds_ratio.round(4),
    "Arah": ["-> Malignant" if v > 0 else "-> Benign" for v in coef],
}).sort_values("Koefisien", ascending=False).reset_index(drop=True)

# ----------------------------------------------------------------------
# 9. Insight 2 -- PR curve data
# ----------------------------------------------------------------------
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)
avg_precision = average_precision_score(y_test, y_pred_prob)

# ----------------------------------------------------------------------
# 10. ROC curve data (for all 4 methods)
# ----------------------------------------------------------------------
roc_data = {}
for name, probs in [
    ("Default", y_def_prob), ("GridSearchCV", y_gs_prob),
    ("RandomizedSearchCV", y_rs_prob), ("Optuna", y_opt_prob),
]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc_score(y_test, probs)}

# ----------------------------------------------------------------------
# 11. Confusion matrix of best model
# ----------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

# ----------------------------------------------------------------------
# 12. Save everything
# ----------------------------------------------------------------------
joblib.dump(best_model, "model/best_model_logreg.pkl")
joblib.dump({
    "mean_benign": mean_benign, "mean_malignant": mean_malignant,
    "std_benign": std_benign, "std_malignant": std_malignant,
    "feature_names": FEATURE_NAMES,
}, "model/benchmark_stats.pkl")

joblib.dump({
    "best_name": best_name,
    "best_metrics": best_metrics,
    "all_results": all_results,
    "baseline_metrics": baseline_metrics,
    "gs_metrics": gs_metrics,
    "gs_best_params": gs.best_params_,
    "rs_metrics": rs_metrics,
    "rs_best_params": rs.best_params_,
    "opt_metrics": opt_metrics,
    "opt_best_params": bp,
    "coef_df": coef_df,
    "pr_curve": {"precisions": precisions, "recalls": recalls, "thresholds": thresholds, "avg_precision": avg_precision},
    "roc_data": roc_data,
    "confusion_matrix": cm,
    "y_test": y_test.values,
    "y_pred": y_pred,
    "y_pred_prob": y_pred_prob,
    "X_test": X_test,
    "X_train": X_train,
    "y_train": y_train.values,
}, "model/results.pkl")

print("Semua artefak berhasil disimpan ke folder model/")
