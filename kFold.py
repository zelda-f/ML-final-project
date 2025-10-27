import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

def perform_kfold_split(data_path, n_splits=5, random_state=42):
    """
    Perform k-fold cross-validation split on the dataset.
    Parameters:
    - data_path: str, path to the CSV file containing the dataset.
    - n_splits: int, number of folds for K-Fold cross-validation.
    - random_state: int, random seed for reproducibility.

    Returns:
    - folds: list of tuples, each containing training and validation indices for each fold.
    """

    # Load the dataset
    df = pd.read_csv(data_path)

    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []
    for train_index, val_index in kf.split(df):
        folds.append((train_index, val_index))

    return folds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    
    args = parser.parse_args()
    
    folds = perform_kfold_split(args.data_path, args.n_splits, args.random_state)
    
    for i, (train_idx, val_idx) in enumerate(folds):
        print(f"Fold {i+1}:")
        print(f"  Training indices: {train_idx}")
        print(f"  Validation indices: {val_idx}")