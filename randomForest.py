import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def train_random_forest_bootstrap(
    df,
    target_col,
    feature_cols=None,
    group_cols=["Problem_id", "Participant_anon"],
    n_bootstrap=10,
    sample_frac=0.8,
    random_state=42,
    test_size=0.2,
    n_estimators=100,
    max_depth=None
):
    """
    Train multiple Random Forest models using bootstrap resampling by group.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target_col : str
        Column name of the target variable.
    feature_cols : list of str, optional
        List of feature columns to use. If None, uses all numeric columns except the target.
    group_cols : list of str
        Columns that define grouping for bootstrapping.
    n_bootstrap : int
        Number of bootstrap samples to create.
    sample_frac : float
        Fraction of unique groups to sample for each bootstrap.
    random_state : int
        Random seed for reproducibility.
    test_size : float
        Fraction of each bootstrap used for testing.
    n_estimators : int
        Number of trees in the Random Forest.
    max_depth : int, optional
        Maximum depth of trees.

    Returns
    -------
    list of dict
        Each dict contains the model, train/test accuracy, and the bootstrap index.
    """

    rng = np.random.default_rng(random_state)

    # Identify feature columns automatically if not specified
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in feature_cols:
            feature_cols.remove(target_col)

    # --- Bootstrapping step ---
    unique_groups = df[group_cols].drop_duplicates()
    n_groups = len(unique_groups)
    n_sample = int(np.ceil(sample_frac * n_groups))

    results = []

    for i in range(n_bootstrap):
        sampled_groups = unique_groups.sample(
            n=n_sample,
            replace=True,
            random_state=rng.integers(1e9)
        )
        merged = df.merge(sampled_groups, on=group_cols, how="inner")

        X = merged[feature_cols]
        y = merged[target_col]

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=rng.integers(1e9)
        )

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=rng.integers(1e9)
        )

        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)

        results.append({
            "bootstrap": i + 1,
            "model": model,
            "train_accuracy": acc_train,
            "test_accuracy": acc_test,
            "n_train": len(X_train),
            "n_test": len(X_test)
        })

        print(f"Bootstrap {i+1}: train_acc={acc_train:.3f}, test_acc={acc_test:.3f}")

    return results

features = pd.read_csv("features.csv")

features = features.dropna(subset=features.select_dtypes(include=[np.number]).columns)

# Train Random Forest models using bootstrap samples
results = train_random_forest_bootstrap(
    df=features,
    target_col="TaskCorrect",  # your classification target
    group_cols=["Problem_id", "Participant_anon"],
    n_bootstrap=5,
    sample_frac=0.8,
    random_state=42
)
