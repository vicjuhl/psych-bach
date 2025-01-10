import pandas as pd
import numpy as np
from collections import defaultdict

from src.data_utils.fields import sorted_features, active_drug
from src.config import n_splits

def aggregate_scores_reg(experiments):
    """Aggregate scores from multiple test folds from the same experiment run."""
    exp_ids, targets, model_names, scrambleds, means, stds = [], [], [], [], [], []

    def new_row(i, target, name, scrambled, scores):
        exp_ids.append(i)
        targets.append(target)
        model_names.append(name)
        scrambleds.append(scrambled)
        means.append(np.mean(scores).item())
        stds.append(np.std(scores).item())
    
    for i, experiment in enumerate(experiments):
        for target, name_dict in experiment.items():
            for name, local_results in name_dict.items():
                # Store true results
                true_scores = local_results["scores"]["true"]
                new_row(i, target, name, False, true_scores)
                # Store scrambled results
                if name == "always_mean": # No scrambled results for always_mean
                    continue
                for scr_scores in local_results["scores"]["scrambled"]:
                    new_row(i, target, name, True, scr_scores)

    return pd.DataFrame({
        "exp_id": exp_ids,
        "target": targets,
        "model_name": model_names,
        "scrambled": scrambleds,
        "mean": means,
        "std": stds
    })

def aggregate_scores_cls(experiments):
    exp_ids, model_names, scrambleds, means, stds = [], [], [], [], []

    def new_row(i, name, scrambled, scores):
        exp_ids.append(i)
        model_names.append(name)
        scrambleds.append(scrambled)
        means.append(np.mean(scores).item())
        stds.append(np.std(scores).item())

    for i, experiment in enumerate(experiments):
        for name, local_results in experiment.items():
            # Store true results
            true_scores = local_results["scores"]["true"]
            new_row(i, name, False, true_scores)
            # Store scrambled results
            for scr_scores in local_results["scores"]["scrambled"]:
                new_row(i, name, True, scr_scores)

    return pd.DataFrame({
        "exp_id": exp_ids,
        "model_name": model_names,
        "scrambled": scrambleds,
        "mean": means,
        "std": stds
    })

def aggregate_feature_importance(results):
    """Aggregate means for k and each feature's importance, grouped by experiment/target/model."""
    exp_ids, targets, model_names, ks = [], [], [], []
    feature_importance_dict = {feature: [] for feature in sorted_features}

    def new_row(i, target, name, k, feat_ranking):
        exp_ids.append(i)
        targets.append(target)
        model_names.append(name)
        ks.append(k)

        feat_ranking_dict = {
            n: s
            for n, s in feat_ranking
        }

        for feature in sorted_features:
            feature_importance_dict[feature].append(feat_ranking_dict.get(feature, 0))

    def add_experiment(i, target, name_dict):
        for name, local_results in name_dict.items():
            if name == "always_mean":
                continue
            for feat_info in local_results["features"]:
                k, feat_ranking = feat_info["k"], feat_info["feat_ranking"]
                new_row(i, target, name, k, feat_ranking)

    # Add regressors' experiments
    for i, reg_exp in enumerate(results["regressors"]):
        for target, name_dict in reg_exp.items():
            add_experiment(i, target, name_dict)
    
    # Add classifiers' experiments
    cls_target = list(active_drug.keys())[0]
    for i, cls_exp in enumerate(results["classifiers"]):
        add_experiment(i, cls_target, cls_exp)

    # Create DataFrame
    data = {
        "exp_id": exp_ids,
        "target": targets,
        "model_name": model_names,
        "k": ks
    }
    data.update(feature_importance_dict)

    return pd.DataFrame(data)

def summarize_params(results):
    """Summarize parameter choices across experiments for each model type and name.
    
    Returns:
        dict: Nested dictionary of form:
        {model_type: {
            model_name: {
                param_name: {param_val: fraction}
            }
        }}
    """
    summary = {
        "regressors": {},
        "classifiers": {}
    }
    
    def add_experiment(model_type, name, params):
        if name not in summary[model_type].keys():
            summary[model_type][name] = {}
        local_summary = summary[model_type][name]
            
        for param_name, param_val in params.items():
            if param_name not in local_summary.keys():
                local_summary[param_name] = {}
            
            val_sum = local_summary[param_name].get(param_val, 0) + 1
            local_summary[param_name][param_val] = val_sum
    
    # Add regressors
    for exp in results["regressors"]:
        for target, name_dict in exp.items():
            for name, local_results in name_dict.items():
                for params in local_results["params"]:
                    add_experiment("regressors", name, params)
    
    # Add classifiers
    for exp in results["classifiers"]:
        for name, local_results in exp.items():
            for params in local_results["params"]:
                add_experiment("classifiers", name, params)
    
    # Convert counts to fractions
    for model_type, name_dict in summary.items():
        for name, param_name_dict in name_dict.items():
            for param_name, val_name_dict in param_name_dict.items():
                val_name_total = sum(val_name_dict.values())
                for val_name in val_name_dict.keys():
                    summary[model_type][name][param_name][val_name] /= val_name_total
    
    return summary

def summarize_confusion_matrices(results):
    """Aggregate confusion matrices from multiple test folds from all experiment runs."""
    # Initialize dictionaries for raw counts and running sums for std calculation
    n_exps = len(results["classifiers"]) * n_splits["outer"]
    confusion_matrices = defaultdict(lambda: np.zeros((n_exps, 2, 2), dtype=float))
    
    # Aggregate classifier results
    for i, cls_exp in enumerate(results["classifiers"]):
        for name, local_results in cls_exp.items():
            for j, conf_matrix in enumerate(local_results["confusion_matrices"]):
                cf_array = np.array(conf_matrix)
                confusion_matrices[name][i * n_splits["outer"] + j] = cf_array / cf_array.sum()
    
    # Convert to regular dictionary
    confusion_matrices = dict(confusion_matrices)
    
    # Calculate normalized matrices and standard deviations
    return {
        name: {
            "means": np.mean(matrices, axis=0),
            "stds": np.std(matrices, axis=0)
        } for name, matrices in confusion_matrices.items()
    }
