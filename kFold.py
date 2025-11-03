import pandas as pd
import numpy as np

def split_kfold(data_path, n_splits=5, random_state=42):

    df = pd.read_csv(data_path)
    group_labels = df.groupby(['Participant_anon', 'Problem_id']).ngroup()

    # Get unique group IDs (each corresponds to one participant-problem "entry")
    unique_groups = np.unique(group_labels)

    # Randomize the order of these groups
    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_groups)

    # Split group IDs into k folds
    group_folds = np.array_split(unique_groups, n_splits)

    # Build folds as DataFrames
    folds = []
    for fold_group_ids in group_folds:
        mask = np.isin(group_labels, fold_group_ids)
        fold_df = df[mask]
        folds.append(fold_df)

    # Print summary
    total_entries = len(unique_groups)
    print(f"Total unique (Participant_anon, Problem_id) entries: {total_entries}\n")

    for i, fold_groups in enumerate(group_folds):
        print(f"Fold {i+1}: {len(fold_groups)} entries")
        print(f"  Entries: {fold_groups}\n")

    return folds


folds = split_kfold("features.csv", n_splits=5)

# Testing more detailed output for each fold
for i, fold_df in enumerate(folds):
    print(f"Fold {i+1}: {len(fold_df)} rows")
    print("  Participants in this fold:", fold_df['Participant_anon'].unique())
    print("  Problem_ids in this fold:", fold_df['Problem_id'].unique())
