import pandas as pd
import numpy as np

def bootstrap_by_group(
    df: pd.DataFrame,
    group_cols=["Problem_id", "Participant_anon"],
    n_bootstrap=100,
    sample_frac=1.0,
    random_state=None
):
    rng = np.random.default_rng(random_state)
    
    unique_groups = df[group_cols].drop_duplicates()
    n_groups = len(unique_groups)
    n_sample = int(np.ceil(sample_frac * n_groups))
    
    bootstraps = []
    for i in range(n_bootstrap):
        sampled_groups = unique_groups.sample(
            n=n_sample,
            replace=True,
            random_state=rng.integers(1e9)
        )
        
        merged = df.merge(sampled_groups, on=group_cols, how="inner")
        bootstraps.append(merged)
    
    return bootstraps


features = pd.read_csv("features.csv")

boot_samples = bootstrap_by_group(
    features,
    group_cols=["Problem_id", "Participant_anon"],
    n_bootstrap=10,
    sample_frac=0.8,
    random_state=42
)

df_boot1 = boot_samples[0]
print(len(df_boot1), "rows in first bootstrap")
print(df_boot1.head())

df_boot2 = boot_samples[1]
print(len(df_boot2), "rows in second bootstrap")
print(df_boot2.head())