import numpy as np
from itertools import product

from src.config import model_params, n_splits, top_k_feats
from src.feature_selection import select_features, vote_feats
from src.utils.util_funcs import argbest
from src.utils.types import f_info

def tune_model(model, train_val_df):
    # Get all possible parameter configurations
    params = model_params[model.__class__.__name__]
    if not params:
        # If no params to tune, just do feature selection on entire train/val set
        (final_k_means, final_feats) = select_features(model, train_val_df)
        final_k = top_k_feats[argbest(model)(final_k_means)]
        return (final_k, final_feats), {}

    # Generate grid of parameter combinations
    param_names = list(params.keys())
    param_values = [params[name]["values"] for name in param_names]
    param_configs = [
        dict(zip(param_names, config))
        for config in product(*param_values)
    ]
    
    # For each config, evaluate across all folds
    mean_scores: list[float] = [] # one score per configuration
    feat_infos: list[f_info] = [] # one feat info per configuration
    for config in param_configs:
        fold_scores = [] # one score per validation fold
        fold_feats = [] # one feat info per validation fold
        
        for val_fold in range(n_splits["middle"]):
            val_mask = train_val_df["val_split"] == val_fold
            train_mask = train_val_df["val_split"] != val_fold
            
            # Select features for this fold
            # Using DEFAULT params
            # Sets features as side-effect
            feat_suggestion = select_features(model, train_val_df[train_mask]) # This line wastes enormous compute, because it uses default params. Should be run in separate for-loop upfront.
            fold_feats.append(feat_suggestion)
            
            # Train and evaluate
            model.set_important_params(final=False, **config)
            model.fit(train_val_df[train_mask])
            fold_scores.append(model.evaluate(train_val_df[val_mask]))
        
        # Store mean score
        mean_scores.append(float(np.mean(fold_scores)))
        feat_infos.append(vote_feats(fold_feats))

    # Extract best the configuration and its associated best feature info
    best_config_idx = argbest(model)(mean_scores)
    final_config = param_configs[best_config_idx]
    final_k_means, final_feats = feat_infos[best_config_idx]
    final_k = top_k_feats[argbest(model)(final_k_means)]
    return (final_k, final_feats), final_config
