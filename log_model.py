import pandas as pd
import numpy as np

from kFold import *

from sklearn.model_selection import GroupKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold
from scipy.stats import chi2_contingency


gaze_df = pd.read_csv("output.csv")

# drop participants with too many nans
participant_nan_thresh = 0.3  # drop participants with >30% missing data
part_nan_frac = gaze_df.isna().mean(axis=1).groupby(gaze_df['Participant_anon']).mean()
to_drop_parts = part_nan_frac[part_nan_frac > participant_nan_thresh].index.tolist()
if to_drop_parts:
    print("Dropping participants with >30% missing data:", to_drop_parts)
    gaze_df = gaze_df[~gaze_df['Participant_anon'].isin(to_drop_parts)].copy()


# build and evaluate a logistic (binary) model, save model + preds

target_col = 'TaskCorrect'
# prepare features: numeric columns only, drop identifiers and the target
drop_like = {"Participant_anon", "Problem_id", "timestamp", "x", "y", 
             "TaskPreError", "X", "Answer_preerr", "Answer", "Problemlist_n", "age",
             "BRT.rt_mean.correct", "BRT.rt_mean.correct.dominant", "BRT.rt_mean.correct.nondominant"}
numeric = gaze_df.select_dtypes(include=[np.number]).columns.tolist()
features = [c for c in numeric if c not in drop_like and c != target_col]


if not features:
    raise ValueError("No numeric feature columns found to train on.")

X = gaze_df[features].copy()
y = gaze_df[target_col].copy()

X = X.fillna(X.median())

groups_all = gaze_df["Participant_anon"].astype(str) + "_" + gaze_df["Problem_id"].astype(str)
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, groups_all, test_size=0.2, random_state=42, stratify=y
)

# normalize
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# # generate k-fold validation
kf = GroupKFold(n_splits=5)
# X_folds = split_kfold(X_train_s)
# y_folds = split_kfold(y_train)

# fit logistic regression

groups = gaze_df["Participant_anon"].astype(str) + "_" + gaze_df["Problem_id"].astype(str)

log_mod = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

scores = cross_val_score(
    log_mod, X_train_s, y_train, cv=kf, groups=groups_train, scoring="accuracy"
)

log_mod.fit(X_train_s, y_train)

# evaluate
y_pred = log_mod.predict(X_test_s)
y_proba = log_mod.predict_proba(X_test_s)[:, 1] if hasattr(log_mod, "predict_proba") else log_mod.decision_function(X_test_s)

print("Classification report:")
print(classification_report(y_test, y_pred))
try:
    auc = roc_auc_score(y_test, y_proba)
    print("ROC AUC:", auc)
except Exception:
    pass
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))


# attach predictions back to original dataframe (for rows that had feature values)
X_all = X.fillna(X.median())  # same preprocessing used above
X_all_s = scaler.transform(X_all)
gaze_df["_logit_pred_proba"] = log_mod.predict_proba(X_all_s)[:, 1]
gaze_df["_logit_pred"] = log_mod.predict(X_all_s)

# persist outputs
gaze_df.to_csv("output_with_preds.csv", index=False)
print("Wrote output_with_preds.csv with _logit_pred and _logit_pred_proba")