from collections import OrderedDict
import numpy as np
from datetime import datetime
import pathlib as pl
import json
import os

from src.utils.random_seeds import generate_random_seeds, partition_seed_iterators

run_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

use_real_data = True
feature_set_suffixes = ("DCC_")
use_all_data = True # WARNING: ONLY TRUE WHEN FULLY READY

n_cores = 10
seeds = generate_random_seeds(False, seed=129)
seed_iterators = partition_seed_iterators(seeds, n_cores)

n_runs = 500
n_scrambles = 1 # per test fold
if n_runs % n_cores != 0:
    print(f"WARNING: n_runs={n_runs} is not divisible by n_cores={n_cores}, fewer runs={n_runs // n_cores * n_cores} will be performed")
    n_runs = n_runs // n_cores * n_cores

n_splits = OrderedDict([
    ("outer", 5),
    ("middle", 4),
    ("inner", 4)
])

top_k_feats = [10, 20, 30, 45]

model_params = OrderedDict([
    # Regressors
    ("MeanGuesser", {}),
    ("LinReg", {}),
    ("ElasticReg", OrderedDict([
        ("alpha", {"default": 1, "values": [0.5, 1, 3, 10]}),
        ("l1_ratio", {"default": 0.5, "values": [0.25, 0.5, 0.75]})
    ])),
    ("RFReg", OrderedDict([
        ('max_depth', {"default": None, "values": [3, 6, None]}),
        ('n_estimators', {"default": 100, "values": [100]}) # A workaround to include n_estimators in param set
    ])),
    # Classifiers
    ("LogisticClassifier", OrderedDict([
        ('C', {'default': 1, "values": [float(v) for v in np.logspace(-3, 3, 7)]})
    ])),
    ("RFClassifier", OrderedDict([
        ('max_depth', {"default": None, "values": [3, 6, None]}),
        ('n_estimators', {"default": 100, "values": [100]}) # A workaround to include n_estimators in param set
    ])),
    ("SVMClassifier", OrderedDict([
        ("C", {"default": 1, "values": [float(v) for v in np.logspace(-3, 3, 7)]})
    ]))
])
n_trees_final = 1000 # Non-tunable parameter n_estimators of RFs will be set to 1000 for final run

data_path = pl.Path("data/")
results_path = data_path / "results"
meta_data = json.load(open(data_path / "metadata.json", "r"))
plot_path = pl.Path(f"plots/{run_date_time}")

# Dummy data configuration
n_main_feats = 5
n_middle_feats = 4
n_relevant_feats = n_main_feats + n_middle_feats
n_noise_feats = 100
n_feats = n_relevant_feats + n_noise_feats
