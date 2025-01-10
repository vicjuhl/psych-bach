import json
import pandas as pd
import pathlib as pl
import os

from src.config import (
    run_date_time,
    data_path,
    use_real_data,
    use_all_data,
    feature_set_suffixes
)
from src.data_utils.fields import five_d_asc_list, sorted_features, labels

results_path = data_path / "results/"

def prepare_real_data(df):
    df = pd.read_csv(pl.Path(os.path.abspath('')).resolve() / "data" / "data.csv")

    df = df.rename(columns={ # TODO: map in fields.py instead
        'sub': 'pid',
        'ses': 'active_drug'
    })
    # Convert ses-lsd/ses-plc to boolean values
    df['active_drug'] = df['active_drug'].map({'ses-lsd': True, 'ses-plc': False})                       

    # Remove rows with NaN values for used feature set
    subs_with_missing = set()
    for col in df.columns:
        if col.startswith(feature_set_suffixes) or col in labels.keys():
            na_rows = df[df[col].isna()]
            if not na_rows.empty:
                for _, row in na_rows.iterrows():
                    subs_with_missing.add((row['pid']))
                    
    # Filter out rows for participant ID's with missing values
    df = df[~df['pid'].isin(subs_with_missing)]
    print(f"Removed ids: {subs_with_missing}")

    # Rescale 5d-ASC columns for data sets 1 and 2:
    df.loc[df["pid"].str.startswith(("sub-d1", "sub-d2")), "OSE"] /= 27
    df.loc[df["pid"].str.startswith(("sub-d1", "sub-d2")), "AIA"] /= 21
    df.loc[df["pid"].str.startswith(("sub-d1", "sub-d2")), "VUS"] /= 18
    df.loc[df["pid"].str.startswith(("sub-d1", "sub-d2")), "AUD"] /= 16
    df.loc[df["pid"].str.startswith(("sub-d1", "sub-d2")), "VIR"] /= 12
    df.loc[df["pid"].str.startswith(("sub-d1", "sub-d2")), "5DASC"] /= 94
    df.loc[df["pid"].str.startswith(("sub-d1", "sub-d2")), "OAV"] /= 66
    # Check that rescaling is within bounds
    for col in ["OSE", "AIA", "VUS", "AUD", "VIR", "5DASC", "OAV"]:
        mini = min(df[col])
        maxi = max(df[col])
        if mini < 0 or maxi > 100:
            raise ValueError(f"Rescaling failed. Column {col} has min {mini}, max {maxi}")
    # Select columns for use
    selected_cols = ['pid', 'active_drug'] + five_d_asc_list + sorted_features
    df = df[selected_cols]

    if not use_all_data:
        # Select data from 20 first unique participant IDs (40 scans)
        twenty_pids = df['pid'].unique()[:20]
        # Check if any of the twenty pids start with sub-d3 and print warning
        d3_pids = [pid for pid in twenty_pids if str(pid).startswith('sub-d3')]
        if d3_pids:
            print(f"WARNING: The following PIDs starting with 'sub-d3' were selected: {d3_pids}")
        # Filter dataframe to only include these 20 participants
        df = df[df['pid'].isin(twenty_pids)]

    print(f"Included {len(df)} scans")
    return df

def load_data(file_path: pl.Path):
    df = pd.read_csv(file_path)
    return prepare_real_data(df)

def dump_results(results, args):
    if use_real_data:
        file_name = f"real_results_{run_date_time}.json"
    else:
        n, drug_sens, sex_sens, age_sens = args['n'], args['drug_sens'], args['sex_sens'], args['age_sens']
        file_name = f"dummy_results_{run_date_time}_n{n}_sens{drug_sens}_{sex_sens}_{age_sens}.json"
    results_path.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(results_path / file_name, "w"))

def load_results(pattern=None):
    """Load results from current run, otherwise load latest results if none were generated during run time."""
    if pattern is None:
        if use_real_data:
            pattern = "real_results"
        else:
            pattern = "dummy_results"
    
    try:
        file_name = f"{pattern}_{run_date_time}.json"
        return json.load(open(results_path / file_name, "r"))
    except FileNotFoundError:
        # Get the latest results file from the directory
        result_files = list(results_path.glob(f"{pattern}_*.json"))
        if not result_files:
            raise FileNotFoundError("No results files found in data directory")
        file_path = max(result_files, key=lambda x: x.stat().st_mtime)
        return json.load(open(file_path, "r"))
    
    