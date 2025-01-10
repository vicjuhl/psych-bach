import numpy as np
import pandas as pd

from src.data_utils.fields import sorted_features, active_drug, five_d_asc_list

def _create_comparison_matrix(df, model_names):
    """Helper function to create a comparison matrix for model performances."""
    n_models = len(model_names)
    name_to_idx = {name: i for i, name in enumerate(model_names)}
    
    # Calculate mean performance for each model across experiments
    model_means = df.groupby('model_name')['mean'].mean()
    
    # Create comparison matrix
    comp_matrix = np.zeros((n_models, n_models))
    for i, model_i in enumerate(model_names):
        for j, model_j in enumerate(model_names):
            comp_matrix[i, j] = model_means[model_i] - model_means[model_j]
            
    return comp_matrix, name_to_idx

def get_cls_comparison_matrix(df_cls_scores):
    """Create comparison matrix for classifier performances."""
    model_names = sorted(df_cls_scores['model_name'].unique())
    return _create_comparison_matrix(df_cls_scores, model_names)

def get_reg_comparison_matrices(df_reg_scores):
    """Create comparison matrices for regressor performances, one per target."""
    # Filter out scrambled results
    df = df_reg_scores[~df_reg_scores['scrambled']]
    model_names = sorted(df['model_name'].unique())
    targets = sorted(df['target'].unique())
    
    # Create comparison matrix for each target
    return {
        target: _create_comparison_matrix(df[df['target'] == target], model_names)
        for target in targets
    }

def _compute_pvalue_against_null(null_dist, mu_actual, greater):
    """Helper function to compute one-tailed p-value comparing actual data against null distribution."""
    if greater:
        # Test if actual value is greater than null distribution
        p_value = (sum(mu_actual <= null_dist)) / (len(null_dist))
    else:
        # Test if actual value is less than null distribution
        p_value = (sum(mu_actual >= null_dist)) / (len(null_dist))
    
    return p_value

def _compute_pairwise_pvalue(data_i, data_j):
    """Helper function to compute pairwise p-value between two models."""
    # Compute difference between scrambled results (null distribution)
    scrambled_i = data_i[data_i['scrambled']].set_index('exp_id')['mean']
    scrambled_j = data_j[data_j['scrambled']].set_index('exp_id')['mean']
    null_diffs = scrambled_i - scrambled_j
    
    # Get mean difference between unscrambled results
    unscr_i = data_i[~data_i['scrambled']]['mean'].mean()
    unscr_j = data_j[~data_j['scrambled']]['mean'].mean()
    observed_diff = unscr_i - unscr_j
    
    # Compute two-tailed p-value
    le = (sum(observed_diff <= null_diffs)) / (len(null_diffs))
    ge = (sum(observed_diff >= null_diffs)) / (len(null_diffs))
    return 2 * min(le, ge)

def compute_cls_pairwise_pvalues(df_cls_scores):
    """
    Compute matrix of pairwise p-values between classifier models using scrambled diffs
    as null distributions.
    """
    # Get unique models and create mapping
    models = sorted(df_cls_scores['model_name'].unique())
    name_to_idx = {name: i for i, name in enumerate(models)}
    n_models = len(models)
    
    # Initialize p-values matrix
    p_values = np.zeros((n_models, n_models))
    
    for i, model_i in enumerate(models):
        for j, model_j in enumerate(models):
            if i == j:
                p_values[i,j] = 1.0
                continue
                
            data_i = df_cls_scores[df_cls_scores['model_name'] == model_i]
            data_j = df_cls_scores[df_cls_scores['model_name'] == model_j]
            p_values[i,j] = _compute_pairwise_pvalue(data_i, data_j)
            
    return p_values, name_to_idx

def compute_reg_pairwise_pvalues(df_reg_scores):
    """
    Compute matrices of pairwise p-values between regressor models using scrambled diffs
    as null distributions, similar to the classifier approach.
    """
    # Get unique models and targets
    models = sorted([m for m in df_reg_scores['model_name'].unique() if m != 'always_mean'])
    targets = sorted(df_reg_scores['target'].unique())
    name_to_idx = {name: i for i, name in enumerate(models)}
    n_models = len(models)
    
    results = {}
    for target in targets:
        target_df = df_reg_scores[df_reg_scores['target'] == target]
        p_values = np.zeros((n_models, n_models))
        
        for i, model_i in enumerate(models):
            for j, model_j in enumerate(models):
                if i == j:
                    p_values[i,j] = 1.0
                    continue
                    
                data_i = target_df[target_df['model_name'] == model_i]
                data_j = target_df[target_df['model_name'] == model_j]
                p_values[i,j] = _compute_pairwise_pvalue(data_i, data_j)
                
        results[target] = p_values
    
    return results, name_to_idx

def get_scramble_p_val(model_data, model_name, target_name, results):
    # Get scrambled results as null distribution
    scrambled_means = model_data[model_data['scrambled']].groupby('exp_id')['mean'].mean()
    
    # Get mean of unscrambled results (single observation)
    mu_actual = model_data[~model_data['scrambled']]['mean'].mean()
    
    # Compute two-sided empirical p-value
    # For classification (Scan Type), we want unscrambled > scrambled
    # For regression, we want unscrambled < scrambled
    if target_name == "Scan Type":
        # Classifiers: larger is better
        p_value = _compute_pvalue_against_null(scrambled_means, mu_actual, greater=True)
        mean_is_better = mu_actual > scrambled_means.mean()
    else:
        # Regressors: smaller is better
        p_value = _compute_pvalue_against_null(scrambled_means, mu_actual, greater=False)
        mean_is_better = mu_actual < scrambled_means.mean()
    
    # Store result
    results.append({
        'model_name': model_name,
        'target': target_name,
        'mean_is_better_than_scr': mean_is_better,
        'p_value': p_value,  # two-sided p-value
        'mu_actual': mu_actual,
        'scrambled_mean': scrambled_means.mean(),
        'mean_diff': mu_actual - scrambled_means.mean()
    })

def compute_scrambled_p_values_reg(df_reg_scores):
    """Compute p-values for regressor models performing statistically significantly differently from their scrambled versions."""
    results = []
    
    # Process each target and model combination
    for target in df_reg_scores['target'].unique():
        target_df = df_reg_scores[df_reg_scores['target'] == target]
        
        # Filter out 'always_mean' model
        for model in target_df[target_df['model_name'] != 'always_mean']['model_name'].unique():
            # Get data for current model
            model_data = target_df[target_df['model_name'] == model]
            
            get_scramble_p_val(model_data, model, target, results)
    
    return pd.DataFrame(results)

def compute_scrambled_p_values_cls(df_cls_scores):
    results = []
    for model in df_cls_scores['model_name'].unique():
        # Get data for current model
        model_data = df_cls_scores[df_cls_scores['model_name'] == model]
        get_scramble_p_val(model_data, model, "Scan Type", results)

    return pd.DataFrame(results)

def compute_regressor_p_values(results_df, greater: bool):
    """
    Compute p-values for each regressor performing better than the 'always_mean' baseline.
    """
    targets = results_df['target'].unique()
    models = results_df['model_name'].unique()
    results = []
    
    for target in targets:
        # Get baseline distribution for this target
        target_mask = results_df['target'] == target
        baseline_mask = results_df['model_name'] == 'always_mean'
        baseline_dist = results_df[target_mask & baseline_mask].set_index('exp_id')['mean']
        baseline_mean = baseline_dist.mean()
        
        for model in models:
            if model == 'always_mean':
                continue
                
            # Get model's mean performance (single observation)
            model_mask = results_df['model_name'] == model
            model_mean = results_df[target_mask & model_mask & ~results_df['scrambled']]['mean'].mean()
            
            # Compute two-tailed p-value
            p_value = _compute_pvalue_against_null(baseline_dist, model_mean, greater)
            
            results.append({
                'target': target,
                'model': model,
                'better_than_mean': model_mean < baseline_mean,
                'p_value': p_value,
                'model_mean': model_mean,
                'mean_guesser_mean': baseline_mean,
                'mean_diff': model_mean - baseline_mean
            })
    
    return pd.DataFrame(results)

def compute_feature_stats(df_feats):
    # Create copy to avoid modifying original
    df = df_feats.copy()
    
    # Identify feature columns (exclude metadata columns)
    
    # Process each row
    for idx in df.index:
        # Get feature values for current row
        features = df.loc[idx, sorted_features].values
        k = df.loc[idx, 'k']
        
        # Find indices of k largest values
        top_k_indices = np.argpartition(features, -k)[-k:]
        
        # Create normalized array (all zeros except top k)
        normalized = np.zeros_like(features)
        top_k_values = features[top_k_indices]
        
        # Normalize top k values to sum to 1
        normalized[top_k_indices] = top_k_values / np.sum(top_k_values)
        
        # Update feature values in DataFrame
        df.loc[idx, sorted_features] = normalized

        # Assertions to verify normalization and correct number of included features
        non_zero_count = (df.loc[idx, sorted_features] != 0).sum()
        feature_sum = df.loc[idx, sorted_features].sum()
        if non_zero_count != k: #and df.loc[idx, 'model_name'] != "elastic_reg":
            print(f"WARNING: Expected {k} non-zero features, but got {non_zero_count} for model {df.loc[idx, 'model_name']}")
        if not np.isclose(feature_sum, 1.0):
            raise ValueError(f"Features should sum to 1, but sum to {feature_sum}")
        
    return df.groupby(['model_name', 'target'])[['k'] + sorted_features].mean().reset_index()

def compute_corr_coefs(data):
    drug_label = list(active_drug.keys())[0]
    labels = [drug_label] + five_d_asc_list
    
    # Create empty DataFrame with feature names as rows and labels as columns
    corr_df = pd.DataFrame(index=sorted_features, columns=labels)
    
    # Calculate correlations for each feature
    for feature in sorted_features:
        # Calculate correlation with active_drug using all data
        corr_coef = np.corrcoef(data[feature], data[drug_label])[0, 1]
        corr_df.loc[feature, drug_label] = corr_coef
        
        # Calculate 5D-ASC correlations using only active drug data
        active_mask = data[drug_label] == True
        active_data = data[active_mask]
        for label in five_d_asc_list:
            corr_coef = np.corrcoef(active_data[feature], active_data[label])[0, 1]
            corr_df.loc[feature, label] = corr_coef
    
    return corr_df