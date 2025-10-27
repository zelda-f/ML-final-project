import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as sk_silhouette

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

gaze_df = add_gaze_count_latency(gaze_df)
gaze_df = add_gaze_count_latency(gaze_df, aoi_col="AOI_LOSide")
print(gaze_df['HOSide_GazeCount'].head())
print(gaze_df['LOSide_GazeCount'].head())

# Add time quartiles

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
