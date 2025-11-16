import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit

def train_random_forest_bootstrap(
    df,
    target_col,
    feature_cols,
    group_cols=["Problem_id", "Participant_anon"],
    n_bootstrap=10,
    sample_frac=0.8,
    random_state=42,
    n_estimators=200,
    max_depth=None
):

    rng = np.random.default_rng(random_state)

    results = []

    # unique groups
    unique_groups = df[group_cols].drop_duplicates()
    n_groups = len(unique_groups)
    n_sample = int(np.ceil(sample_frac * n_groups))

    for i in range(n_bootstrap):
        sampled = unique_groups.sample(
            n=n_sample,
            replace=True,
            random_state=rng.integers(1e9)
        )

        # subset df to those groups
        merged = df.merge(sampled, on=group_cols, how="inner")
        df_boot = merged.copy()

        # assign each row a composite group label
        groups = df_boot[group_cols].astype(str).agg("_".join, axis=1)

        X = df_boot[feature_cols]
        y = df_boot[target_col]

        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=rng.integers(1e9)
        )
        train_idx, test_idx = next(splitter.split(X, y, groups))

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=rng.integers(1e9)
        )

        model.fit(X_train, y_train)

        # evaluate
        acc_train = accuracy_score(y_train, model.predict(X_train))
        acc_test = accuracy_score(y_test, model.predict(X_test))

        results.append({
            "bootstrap": i+1,
            "model": model,
            "train_accuracy": acc_train,
            "test_accuracy": acc_test,
            "feature_importances": model.feature_importances_,
        })

        print(f"Bootstrap {i+1}: Train={acc_train:.3f}, Test={acc_test:.3f}")

    return results


features = pd.read_csv("output.csv")

features = features.dropna(subset=features.select_dtypes(include=[np.number]).columns)

feature_cols = ["HOSide_GazeCount","HOSide_FirstLook",
                "LOSide_GazeCount","LOSide_FirstLook",
                "peak_HOO_quartile","FirstAOI_HOSide","FirstAOI_LOSide","FirstAOI_Time",
                "gaze_covariance","HOO_AvgLookTime", "HOO_pos_R",
                "HOO_pos_L", "BACKWARDSSPATIALSPAN.object_count_span.overall", 
                "FLANKER.rcs.overall"]


# Train Random Forest models using bootstrap samples
results = train_random_forest_bootstrap(
    df=features,
    feature_cols=feature_cols,
    target_col="TaskCorrect",  # your classification target
    group_cols=["Problem_id", "Participant_anon"],
    n_bootstrap=5,
    sample_frac=0.8,
    random_state=42
)

# aggregate most important features
best_features = np.array([r["feature_importances"] for r in results])

mean_imp = best_features.mean(axis=0)
std_imp = best_features.std(axis=0)

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "mean_importance": mean_imp,
    "std_importance": std_imp
}).sort_values("mean_importance", ascending=False)

print(importance_df)
