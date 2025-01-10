import numpy as np
import pandas as pd

from src.config import meta_data

def list_of_iter_rows(lst, whitespace=0):
    string = "\n"
    for item in lst:
        string += f"{' ' * whitespace}{item}\n"
    return string

def print_dict(d, indent=0):
    """Pretty print a dictionary with proper indentation regardless of nesting depth."""
    for key, value in d.items():
        print(" " * indent + str(key) + ":", end="")
        if isinstance(value, dict):
            print()
            print_dict(value, indent + 2)
        elif isinstance(value, np.ndarray):
            print()
            print(value)
        else:
            print(" " + str(value))

def print_reg_scores_stats(df):
    """Print mean and standard deviation of mean scores for each model and target.
    
    Args:
        df (pd.DataFrame): DataFrame containing regression scores
    """
    print("\nRegression scores statistics:")
    # Group by model and target to calculate statistics
    stats = df.groupby(['model_name', 'target'])['mean'].agg(['mean', 'std']).round(3)
    
    # Print statistics for each combination
    for (model, target) in stats.index:
        mean, std = stats.loc[(model, target)]
        print(f"\n{model} - {meta_data[target]['LongName']}:")
        print(f"  Score: {mean:.3f} +/- {std:.3f}")

def print_cls_scores_stats(df):
    """Print mean and standard deviation of mean scores for each model.
    
    Args:
        df (pd.DataFrame): DataFrame containing classification scores
    """
    print("\nClassification scores statistics:")
    # Group by model to calculate statistics
    stats = df.groupby('model_name')['mean'].agg(['mean', 'std']).round(3)
    
    # Print statistics for each model
    for model in stats.index:
        mean, std = stats.loc[model]
        print(f"\n{model}:")
        print(f"  Score: {mean:.3f} +/- {std:.3f}")

def print_comparison_matrix(matrix, name_mapping, title, target=None):
    """Pretty print a comparison matrix.
    """
    if target:
        title += f" for\n{meta_data[target]["LongName"]}"
    print(f"\n{title}")
    print("(row model - column model)")
    
    # Get sorted model names for consistent ordering
    model_names = sorted(name_mapping.keys())
    
    # Print column headers with proper alignment
    print("\nModel".ljust(20), end="")
    for name in model_names:
        print(f"{name:>20}", end="")
    print()
    
    # Print rows
    for row_name in model_names:
        print(f"{row_name:<15}", end="")
        for col_name in model_names:
            i, j = name_mapping[row_name], name_mapping[col_name]
            print(f"{matrix[i,j]:>20.3f}", end="")
        print()

def print_feature_stats(df_feat_imps, k):
    """Print feature importance rankings for each target.
    
    Args:
        df_feat_imps (pd.DataFrame): DataFrame containing feature importance scores
    """
    print(f"\nFeature importance rankings by target (top {k} features):")
    # Remove metadata columns before calculating means
    meta_cols = ['model_name', 'k', 'target']
    feature_cols = [col for col in df_feat_imps.columns if col not in meta_cols]
    
    # Convert feature columns to numeric, coercing errors to NaN
    df_feat_imps[feature_cols] = df_feat_imps[feature_cols].apply(pd.to_numeric, errors='coerce')
    
    # Calculate means only for numeric feature columns, grouped by target
    target_means = df_feat_imps.groupby('target')[feature_cols].mean().reset_index()
    
    # Process each target
    for target in target_means['target'].unique():
        print(f"\n{meta_data[target]['LongName']}:")
        
        # Get feature scores for this target
        target_row = target_means[target_means['target'] == target].iloc[0]
        feat_scores = target_row[feature_cols].values
        
        # Get non-zero features and their indices
        nonzero_mask = feat_scores > 0
        nonzero_indices = np.where(nonzero_mask)[0]
        nonzero_scores = feat_scores[nonzero_mask]
        
        if len(nonzero_scores) == 0:
            print("  WARNING: All features have zero importance")
            continue
            
        # Sort by importance score descending
        sorted_indices = np.argsort(nonzero_scores)[::-1]
        
        # Print each feature and its score
        for idx in sorted_indices[:k]:
            feat_name = feature_cols[nonzero_indices[idx]]
            score = nonzero_scores[idx]
            print(f"  {feat_name}: {score:.4f}")

def print_k_stats(df_feat_imps):
    """Print distribution of k values for each model and target combination.
    
    Args:
        df_feat_imps (pd.DataFrame): DataFrame containing feature importance scores
        with columns ['model_name', 'k', 'target', 'exp_id']
    """
    print("\nFeature count (k) distribution by model and target:")
    
    # Count total feature selections for each model/target combination
    total_selections = df_feat_imps.groupby(['model_name', 'target']).size()
    
    # Count occurrences of each k value
    k_counts = df_feat_imps.groupby(['model_name', 'target', 'k']).size()
    
    # Print distributions for each combination
    for model in df_feat_imps['model_name'].unique():
        print(f"\n{model}:")
        for target in df_feat_imps[df_feat_imps['model_name'] == model]['target'].unique():
            total = total_selections[model, target]
            print(f"  {meta_data[target]['LongName']}:")
            k_values = sorted(df_feat_imps[
                (df_feat_imps['model_name'] == model) & 
                (df_feat_imps['target'] == target)
            ]['k'].unique())
            for k in k_values:
                count = k_counts.get((model, target, k), 0)
                ratio = count / total
                print(f"    k={int(k)}: {ratio:.3f} ({count}/{total})")

