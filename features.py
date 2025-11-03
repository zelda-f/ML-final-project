import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as sk_silhouette
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
from scipy.stats import chi2_contingency

gaze_df = pd.read_csv('features.csv')

def add_gaze_count_latency(all_AOI_hit, aoi_col="AOI_HOSide"):
    AOI = aoi_col[4:] if len(aoi_col) > 4 else aoi_col
    print("AOI:", AOI)

    def summarize_group(gaze_df):
        hits = gaze_df[gaze_df[aoi_col]]
        gaze_count = len(hits)
        gaze_first = hits['timestamp'].min() if gaze_count > 0 else np.nan
        return pd.Series({
            f"{AOI}_GazeCount": gaze_count,
            f"{AOI}_FirstLook": gaze_first
        })

    summary = (
        all_AOI_hit
        .groupby(['Participant_anon', 'Problem_id'], group_keys=False)
        .apply(summarize_group)
        .reset_index()
    )

    all_AOI_hit = all_AOI_hit.merge(summary, on=['Participant_anon', 'Problem_id'], how='left')
    return all_AOI_hit

def add_avg_HOO_time(df, aoi_col="AOI_HOO"):
    AOI = aoi_col[4:] if len(aoi_col) > 4 else aoi_col 
    avg_col = f"{AOI}_AvgLookTime"

    def summarize_group(group):
        
        group = group.sort_values("timestamp").reset_index(drop=True)

        group["prev"] = group[aoi_col].shift(1, fill_value=False)
        group["look_change"] = (group[aoi_col] != group["prev"])

        group["look_id"] = (group["look_change"] & group[aoi_col]).cumsum() * group[aoi_col]

        looks = (
            group[group[aoi_col]]
            .groupby("look_id")
            .agg(start=("timestamp", "min"), end=("timestamp", "max"))
        )

        looks["duration"] = looks["end"] - looks["start"]
        total_time = looks["duration"].sum()
        num_looks = len(looks)
        avg_time = total_time / num_looks if num_looks > 0 else np.nan

        return pd.Series({avg_col: avg_time})

    summary = (
        df.groupby(["Participant_anon", "Problem_id"], group_keys=False)
        .apply(summarize_group)
        .reset_index()
    )

    df = df.merge(summary, on=["Participant_anon", "Problem_id"], how="left")
    return df


gaze_df = add_gaze_count_latency(gaze_df)
gaze_df = add_gaze_count_latency(gaze_df, aoi_col="AOI_LOSide")
print(gaze_df['HOSide_GazeCount'].head())
print(gaze_df['LOSide_GazeCount'].head())

def assign_quartile(group):
    """
    Adds column "time_quartile" which is an int 1-4 that represents the time quartile of the timestamp column
    """
    total_time = group['task_resp.rt'].iloc[0]
    q1, q2, q3 = total_time / 4, total_time / 2, 3 * total_time / 4

    def get_quartile(ts):
        if ts <= q1:
            return 1
        elif ts <= q2:
            return 2
        elif ts <= q3:
            return 3
        else:
            return 4

    group['time_quartile'] = group['timestamp'].apply(get_quartile)
    return group

gaze_df = gaze_df.groupby(['Participant_anon', 'Problem_id'], group_keys=False).apply(assign_quartile)

print(gaze_df['time_quartile'].head())

# add peak proportion quartile

def peak_prop_quart(df, AOI_col):
    # Compute mean hit per quartile within each participant/problem
    quart_means = (
        df.groupby(['Participant_anon', 'Problem_id', 'time_quartile'])[AOI_col]
        .mean()
        .reset_index()
    )

    # Find quartile with the max mean per participant/problem
    peak_quart = (
        quart_means.loc[
            quart_means.groupby(['Participant_anon', 'Problem_id'])[AOI_col].idxmax(),
            ['Participant_anon', 'Problem_id', 'time_quartile']
        ]
        .rename(columns={'time_quartile': 'peak_HOO_quartile'})
    )

    # Merge the peak quartile info back into the main dataframe
    df = df.merge(peak_quart, on=['Participant_anon', 'Problem_id'], how='left')

    return df

gaze_df = peak_prop_quart(gaze_df, "AOI_HOSide")
print(gaze_df['peak_HOO_quartile'].mean())
# Take the covariance of gaze points in x and y per participant/problem
def add_gaze_covariance(gaze_df):
    def summarize_covariance(group):
        cov = group[['x', 'y']].cov().iloc[0, 1]
        return pd.Series({
            'gaze_covariance': cov
        })

    covariance_summary = (
        gaze_df
        .groupby(['Participant_anon', 'Problem_id'], group_keys=False)
        .apply(summarize_covariance)
        .reset_index()
    )

    gaze_df = gaze_df.merge(covariance_summary, on=['Participant_anon', 'Problem_id'], how='left')
    return gaze_df

gaze_df = add_gaze_covariance(gaze_df)
print(gaze_df[['gaze_covariance']].head())

def silhouette_score(cluster_range=range(2, 7), coord_candidates=None, sample_size=5000, plot=False):
    """
    Compute silhouette scores for different k (number of clusters) per time_quartile.
    Returns (results_df, best_per_quartile_df).
    - cluster_range: iterable of ints (k values) to try, default 2..6
    - coord_candidates: list of possible coordinate column names to look for (will pick first two found)
    - sample_size: max number of points to sample per quartile (to limit compute)
    - plot: if True, show a simple line plot of silhouette vs k for each quartile
    """

    # detect coordinate columns
    coord_candidates = [
       "x", "y"
    ]
    found = [c for c in coord_candidates if c in gaze_df.columns]
    if len(found) < 2:
        raise ValueError(f"Could not find two coordinate columns. Candidates checked: {coord_candidates}. Found: {found}")

    coords = found[:2]
    results = []

    quartiles = sorted(gaze_df['time_quartile'].dropna().unique())
    for q in quartiles:
        sub = gaze_df[gaze_df['time_quartile'] == q]
        X = sub[coords].dropna()
        if len(X) < 2:
            # not enough points to cluster
            continue
        if len(X) > sample_size:
            X = X.sample(sample_size, random_state=42)
        X_vals = X.values
        for k in cluster_range:
            if k < 2:
                continue
            if k >= len(X_vals):
                # cannot have more clusters than samples
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = km.fit_predict(X_vals)
                try:
                    s = sk_silhouette(X_vals, labels)
                except Exception:
                    s = np.nan
            results.append({
                "time_quartile": q,
                "n_clusters": k,
                "silhouette": float(s)
            })

    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("No silhouette scores computed (not enough data).")
        return res_df, pd.DataFrame()

    best_idx = res_df.groupby("time_quartile")["silhouette"].idxmax()
    best_df = res_df.loc[best_idx].reset_index(drop=True)

    if plot:
        import matplotlib.pyplot as plt
        for q in sorted(res_df["time_quartile"].unique()):
            subset = res_df[res_df["time_quartile"] == q].sort_values("n_clusters")
            plt.plot(subset["n_clusters"], subset["silhouette"], marker="o", label=f"quartile {q}")
        plt.xlabel("n_clusters")
        plt.ylabel("silhouette score")
        plt.legend()
        plt.tight_layout()
        plt.show()

    print("Best number of clusters per quartile:")
    print(best_df)
    return res_df, best_df
    
res_df, best_df = silhouette_score((2, 10), None, 5000, True)

print(best_df)
print(res_df)



gaze_df = add_avg_HOO_time(gaze_df)
print("Average HOO Look Time:")
print(gaze_df[['Participant_anon', 'Problem_id', 'HOO_AvgLookTime']])
gaze_df.to_csv("output.csv", index=False)

# build and evaluate a logistic (binary) model, save model + preds

# 1) find a binary target column
candidate_names = [
    "task_resp.corr", "correct", "is_correct", "accuracy",
    "response_correct", "outcome", "success"
]
target_col = None
for name in candidate_names:
    if name in gaze_df.columns and gaze_df[name].nunique(dropna=True) == 2:
        target_col = name
        break

# fallback: any numeric/bool column with exactly 2 unique non-null values
if target_col is None:
    for col in gaze_df.columns:
        if gaze_df[col].nunique(dropna=True) == 2:
            target_col = col
            break

if target_col is None:
    raise ValueError("No binary target column found. Add a binary target (0/1) or one of: " + ", ".join(candidate_names))

print("Using target:", target_col)

# 2) prepare features: numeric columns only, drop identifiers and the target
drop_like = {"Participant_anon", "Problem_id", "timestamp"}
numeric = gaze_df.select_dtypes(include=[np.number]).columns.tolist()
features = [c for c in numeric if c not in drop_like and c != target_col]

if not features:
    raise ValueError("No numeric feature columns found to train on.")

X = gaze_df[features].copy()
y = gaze_df[target_col].copy()

# 3) coerce non-numeric binary targets to 0/1
if not pd.api.types.is_numeric_dtype(y):
    y, uniques = pd.factorize(y)
else:
    # ensure binary is 0/1
    vals = sorted(y.dropna().unique())
    if set(vals) == {False, True} or set(vals) == {0, 1}:
        y = y.astype(int)
    else:
        # map two unique values to 0/1 deterministically
        mapping = {vals[0]: 0, vals[1]: 1} if len(vals) == 2 else {}
        if mapping:
            y = y.map(mapping)

# 4) simple preprocessing: drop cols with too many nans, fill remaining with median
thresh = int(0.5 * len(X))
to_drop = [c for c in X.columns if X[c].isna().sum() > thresh]
if to_drop:
    print("Dropping cols with >50% missing:", to_drop)
    X = X.drop(columns=to_drop)
X = X.fillna(X.median())

# 5) train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6) scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 7) fit logistic regression
clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
clf.fit(X_train_s, y_train)

# 8) evaluate
y_pred = clf.predict(X_test_s)
y_proba = clf.predict_proba(X_test_s)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test_s)

print("Classification report:")
print(classification_report(y_test, y_pred))
try:
    auc = roc_auc_score(y_test, y_proba)
    print("ROC AUC:", auc)
except Exception:
    pass
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# 9) save model, scaler, and feature list
joblib.dump({"model": clf, "scaler": scaler, "features": X.columns.tolist(), "target": target_col}, "logistic_model.joblib")
print("Saved logistic_model.joblib")

# 10) attach predictions back to original dataframe (for rows that had feature values)
X_all = X.fillna(X.median())  # same preprocessing used above
X_all_s = scaler.transform(X_all)
gaze_df["_logit_pred_proba"] = clf.predict_proba(X_all_s)[:, 1]
gaze_df["_logit_pred"] = clf.predict(X_all_s)

test_table = pd.crosstab(gaze_df['TaskCorrect'], gaze_df['_logit_pred'])
chi, p, dof, ef = chi2_contingency(test_table)
print("pval: ", p)

# 11) persist outputs
gaze_df.to_csv("output_with_preds.csv", index=False)
print("Wrote output_with_preds.csv with _logit_pred and _logit_pred_proba")