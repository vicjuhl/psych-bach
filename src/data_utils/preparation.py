import pandas as pd

from src.data_utils.fields import features
from src.data_generation.dummy_data import gen_dummy_set
from src.config import data_path, use_real_data, use_all_data
from src.data_utils.imports_exports import load_data

def whiten(df: pd.DataFrame, train_mask: pd.DataFrame, test_mask: pd.DataFrame):
    """Whiten both df[train_mask] and df[test_mask] based on metrics from train_mask"""
    for col in features.keys():
        # Get mean and std of training set
        mu_train = df[train_mask][col].mean()
        std_train = df[train_mask][col].std()
        if std_train == 0:
            print(f"Warning: Feature {col} has zero standard deviation")
            continue
        # Correct entire data based on the training statistics
        df[col] = (df[col] - mu_train) / std_train
    return df[train_mask], df[test_mask]

def whiten_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Whitens (standardizes) all feature columns in the dataset without considering splits.
    Returns a copy of the dataframe with whitened features.
    """
    df_copy = df.copy()
    for col in features.keys():
        mu = df_copy[col].mean()
        std = df_copy[col].std()
        if std == 0:
            print(f"Warning: Feature {col} has zero standard deviation")
            continue
        df_copy[col] = (df_copy[col] - mu) / std
    return df_copy

def active_only(df: pd.DataFrame):
    return df[df['active_drug'] == True]

def get_data(args):
    """
    Get the data from the data folder or generate dummy data if gen_data is True.
    """
    if use_real_data:
        df = load_data(data_path / "data.csv")
    else: # dummy data
        n = args.get('n', 1000)
        drug_sens = args.get('drug_sens', 0.5)
        sex_sens = args.get('sex_sens', 0.5)
        age_sens = args.get('age_sens', 0.5)
        if args["gen_data"]:
            gen_dummy_set(n, drug_sens, sex_sens, age_sens, data_path)
        df = pd.read_csv(data_path / f"dummy_n{n}_sens{drug_sens}_{sex_sens}_{age_sens}.csv")
    return df
