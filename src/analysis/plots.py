import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

from src.config import plot_path, meta_data, use_real_data
from src.data_utils.fields import active_drug, five_d_asc_list, long_model_names, sorted_features
from src.analysis.statistics import compute_corr_coefs

def qq_plot(group, target, model_name, save_path, args):
    # Get scores from the group
    scores = group['mean'].values
    
    # Create QQ plot using scipy.stats
    stats.probplot(scores, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot for {model_name} predicting {target}", fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    
    # Save plot as PNG
    save_path.mkdir(parents=True, exist_ok=True)
    file_name = f"qq_plot_{model_name}_{target}"
    if not use_real_data:
        file_name += f"_n{args['n']}_sens{args['drug_sens']}_{args['sex_sens']}_{args['age_sens']}"
    plt.savefig(save_path / f"{file_name}.png")
    plt.close()

def qq_plots_reg(score_df, args):
    # Group by target and model_name, only use unscrambled results
    grouped = score_df[~score_df['scrambled']].groupby(['target', 'model_name'])
    
    for (target, model_name), group in grouped:
        qq_plot(group, target, model_name, plot_path / "qq_plots" / "regressors/", args)

def qq_plots_cls(score_df, args):
    # Group by model_name only since target is fixed for classifiers
    grouped = score_df[~score_df['scrambled']].groupby('model_name')
    
    for model_name, group in grouped:
        # Get target name from fields
        target = list(active_drug.keys())[0]
        qq_plot(group, target, model_name, plot_path / "qq_plots" / "classifiers/", args)

def plot_score_distributions_reg(df_reg_scores, args):
    # Plot histograms comparing scrambled vs unscrambled regression scores
    save_dir = plot_path / "score_distributions" / "regressors"
    save_dir.mkdir(parents=True, exist_ok=True)

    grouped_reg = df_reg_scores.groupby(['target', 'model_name'])
    for (target, model_name), group in grouped_reg:
        plt.title(f"Score Distribution: {long_model_names[model_name]}\npredicting {meta_data[target]['LongName']}", 
                 fontsize=14)
        plt.xlabel("MSE", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.tick_params(axis='both', labelsize=12)
        if model_name == "always_mean":
            # Plot histogram for mean guesser
            plt.hist(group['mean'], bins=20, density=True, alpha=0.7, color='blue', label='Mean Guesser Distribution')
            
            # Add vertical lines for other models' means
            other_models_df = df_reg_scores[
                (df_reg_scores['target'] == target) & 
                (~df_reg_scores['scrambled']) & 
                (df_reg_scores['model_name'] != 'always_mean')
            ]
            colors = {'elastic_reg': 'red', 'lin_reg': 'green', 'random_forest_regressor': 'orange'}
            for (name, model_data), v_pos in zip(other_models_df.groupby('model_name'), [0.9, 0.8, 0.7]):
                mean_val = model_data['mean'].mean()
                std_val = model_data['mean'].std()
                
                # Vertical line for mean
                plt.axvline(
                    mean_val,
                    color=colors[name],
                    linestyle='--',
                    label=f'{long_model_names[name]} mean',
                    linewidth=2
                )
                
                # Add horizontal line for ±1 standard deviation
                y_pos = plt.gca().get_ylim()[1] * v_pos  # Position at 90% of plot height
                plt.hlines(
                    y=y_pos,
                    xmin=mean_val - std_val,
                    xmax=mean_val + std_val,
                    color=colors[name], alpha=0.5
                )
            plt.legend(loc='center right')
        else:
            # Get scrambled and unscrambled scores
            scrambled = group[group['scrambled']]['mean']
            unscrambled = group[~group['scrambled']]['mean']
            
            # Calculate common range for bins
            min_val = min(scrambled.min(), unscrambled.min())
            max_val = max(scrambled.max(), unscrambled.max())
            bins = np.linspace(min_val, max_val, 21)  # 20 bins (needs 21 edges)
            
            # Plot normalized histograms with shared bins
            plt.hist(unscrambled, bins=bins, density=True, alpha=0.5, color='blue', label='Actual')
            plt.axvline(unscrambled.mean(), color='blue', label=f'Actual mean')
            plt.hist(scrambled, bins=bins, density=True, alpha=0.5, color='red', label='Permuted')
            plt.legend()
        
        # Save plot
        file_name = f"score_dist_{model_name}_{target}"
        if not use_real_data:
            file_name += f"_n{args['n']}_sens{args['drug_sens']}_{args['sex_sens']}_{args['age_sens']}"
        plt.tight_layout()
        plt.savefig(save_dir / f"{file_name}.png")
        plt.close()

def plot_score_distributions_cls(df_cls_scores, args):
    grouped_cls = df_cls_scores.groupby('model_name')
    save_dir = plot_path / "score_distributions" / "classifiers"
    save_dir.mkdir(parents=True, exist_ok=True)

    for model_name, group in grouped_cls:
        plt.title(f"Score Distribution: {long_model_names[model_name]}", fontsize=14)
        plt.xlabel("F1", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.tick_params(axis='both', labelsize=12)
        # Get scrambled and unscrambled scores
        scrambled = group[group['scrambled']]['mean']
        unscrambled = group[~group['scrambled']]['mean']
        
        # Calculate common range for bins
        min_val = min(scrambled.min(), unscrambled.min())
        max_val = max(scrambled.max(), unscrambled.max())
        bins = np.linspace(min_val, max_val, 21)  # 20 bins (needs 21 edges)
        
        # Plot normalized histograms with shared bins
        plt.hist(unscrambled, bins=bins, density=True, alpha=0.6, color='blue', label='Actual')
        plt.axvline(unscrambled.mean(), color='blue', label=f'Actual mean', linewidth=2)
        plt.hist(scrambled, bins=bins, density=True, alpha=0.6, color='red', label='Permuted')
        plt.legend(fontsize=12)
        file_name = f"score_dist_{model_name}"
        if not use_real_data:
            file_name += f"_n{args['n']}_sens{args['drug_sens']}_{args['sex_sens']}_{args['age_sens']}"
        plt.savefig(save_dir / f"{file_name}.png")
        plt.close()

def plot_individual_feat_importance(model_scores, feature_cols, models, save_dir, target=None):
    for model in models:
        # Create individual plot for each model
        sorted_indices = np.argsort(model_scores[model])
        sorted_scores = model_scores[model].iloc[sorted_indices]
        sorted_features = model_scores[model].index[sorted_indices]
        sorted_feat_names = [str.replace(f[4:], "_", " – ") for f in sorted_features]
        
        plt.figure(figsize=(10, max(8, len(feature_cols)/3)))
        title = f"Feature Importance Ranking:\n{long_model_names[model]}"
        if target:
            title += f"\npredicting {meta_data[target]['LongName']}"
        plt.title(title, fontsize=14, pad=20)
        
        y_pos = np.arange(len(sorted_features))
        plt.barh(y_pos, sorted_scores)
        
        plt.yticks(y_pos, sorted_feat_names, fontsize=12)
        plt.xlabel('Importance Score', fontsize=12)
        plt.tick_params(axis='x', labelsize=12)
        
        plt.tight_layout()
        plt.savefig(save_dir / f"feature_importance_{model}.png")
        plt.close()

def plot_feature_importance_comparison(model_scores, feature_cols, models, save_dir, target=None, corr_coefs=None):
    sorted_indices = np.argsort(model_scores[models[0]])
    sorted_features = model_scores[models[0]].index[sorted_indices]
    sorted_feat_names = [str.replace(f[4:], "_", " – ") for f in sorted_features]
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(feature_cols)/3)))
    
    bar_width = 0.2  # Reduced from 0.25 to accommodate fourth bar
    y_pos = np.arange(len(sorted_features))
    
    # Plot model importance scores
    for i, (model, color) in enumerate(zip(models, ['blue', 'green', 'red'])):
        scores = [model_scores[model][feat] for feat in sorted_features]
        ax.barh(
            y_pos + i*bar_width,
            scores,
            bar_width, 
            label=long_model_names[model],
            color=color,
            alpha=0.7
        )
    
    # Add correlation coefficients if provided
    if corr_coefs is not None:
        # For classifiers, use the first drug column if target is None
        target_col = target if target is not None else list(active_drug.keys())[0]
        corr_scores = [corr_coefs.loc[feat, target_col] for feat in sorted_features]
        
        # Normalize correlation coefficients to sum to 1 while preserving sign
        abs_sum = sum(abs(score) for score in corr_scores)
        norm_corr_scores = [score / abs_sum for score in corr_scores]
        
        ax.barh(
            y_pos + 3*bar_width,
            norm_corr_scores,
            bar_width,
            label='Correlation',
            color='orange',
            alpha=0.7
        )
    
    ax.set_yticks(y_pos + 1.5*bar_width)  # Center ticks between bars
    ax.set_yticklabels(sorted_feat_names, fontsize=12)
    ax.set_xlabel(f'Importance Score / Normalized Correlation (max(|ρ|) = {max([abs(rho) for rho in corr_scores]):.2f})', 
                 fontsize=12)
    ax.tick_params(axis='x', labelsize=12)
    
    title = 'Feature Importance Comparison'
    if target:
        title += f"for prediction {meta_data[target]['LongName']}"
    else:
        # For classifiers, use the drug name
        drug_name = list(active_drug.keys())[0]
        title += f"\npredicting {meta_data[drug_name]['LongName']}"
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_dir / "feature_importance_comparison.png")
    plt.close()

def plot_feat_stats(feat_stats, corr_coefs):
    # Define model groups
    # First entry in each list defines sorting order
    classifier_models = ['logistic_regression', 'svm', 'random_forest_classifier']
    regressor_models = ['elastic_reg', 'random_forest_regressor', 'lin_reg']
    
    # Get feature columns
    feature_cols = [
        col for col in feat_stats.columns 
        if col not in ['model_name', 'k', 'target']
    ]
    
    # Process classifiers and regressors separately
    for model_type, models in [('classifiers', classifier_models), 
                             ('regressors', regressor_models)]:
        
        # Create save directory
        save_dir = plot_path / "feature_importance" / model_type
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter for relevant models
        filtered_stats = feat_stats[feat_stats['model_name'].isin(models)]
        
        # Process each target separately for regressors
        targets = filtered_stats['target'].unique() if model_type == 'regressors' else [None]
        
        for target in targets:
            target_stats = filtered_stats
            target_dir = save_dir
            
            if target is not None:
                target_stats = filtered_stats[filtered_stats['target'] == target]
                target_dir = save_dir / target
                target_dir.mkdir(exist_ok=True)
            
            # Create dictionary to store feature scores for each model
            model_scores = {}
            for model in models:
                group = target_stats[target_stats['model_name'] == model]
                if not group.empty:
                    model_scores[model] = group[feature_cols].mean()
            
            if model_scores:
                # Create individual plots and comparison plot
                plot_individual_feat_importance(model_scores, feature_cols, models, target_dir, target)
                plot_feature_importance_comparison(model_scores, feature_cols, models, target_dir, target, corr_coefs)

def plot_feature_target_scatter(data):
    # Separate features and targets
    drug_col = list(active_drug.keys())[0]
    target_cols = [drug_col] + five_d_asc_list
    
    # Randomly select 3 features
    np.random.seed(420)  # Set seed for reproducibility
    random_features = np.random.choice(sorted_features, size=4, replace=False)
    
    # Create a grid of scatter plots (6 targets x 3 features)
    fig, axes = plt.subplots(len(target_cols), len(random_features), figsize=(13, 14))
    fig.suptitle('Feature-Target Relationships (Random Sample)', fontsize=20, y=0.96)
    
    # Create scatter plots
    for i, target in enumerate(target_cols):
        for j, feature in enumerate(random_features):
            ax = axes[i, j]
            
            plot_data = data.copy()
            # Filter for active drug rows only when plotting 5D-ASC scores
            if target in five_d_asc_list:
                plot_data = data[data[drug_col]]
            
            # Create scatter plot
            ax.scatter(plot_data[feature], plot_data[target], alpha=0.5, s=15)
            
            # Add labels and title with larger font sizes
            if i == len(target_cols)-1:  # Bottom row
                ax.set_xlabel(feature[4:].replace('_', ' – '), fontsize=12)
            if j == 0:  # First column
                ax.set_ylabel(target, fontsize=12)
            if i == 0:  # Top row
                ax.set_title(feature[4:].replace('_', ' – '), fontsize=12)
            
            # Increase tick label size
            ax.tick_params(axis='both', labelsize=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_dir = plot_path / "feature_target_relationships"
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "feature_target_scatter.png")
    plt.close()

def plot_5d_asc_distributions(data):
    save_dir = plot_path / "5d_asc_dist"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create histogram for each 5D-ASC dimension
    for target in five_d_asc_list:
        plt.figure(figsize=(8, 6))
        plt.xlabel("Score", fontsize=16)
        plt.ylabel("Density", fontsize=16)
        plt.tick_params(axis='both', labelsize=16)
        
        # Plot histogram with density and add KDE
        scores = data[data[list(active_drug.keys())[0]]][target]
        plt.hist(scores, bins=20, density=True, alpha=0.7, label='Histogram')
        
        # Add mean and std lines
        mean_val = scores.mean()
        std_val = scores.std()
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        plt.axvline(mean_val - std_val, color='green', linestyle=':', label=f'±1 SD: {std_val:.2f}')
        plt.axvline(mean_val + std_val, color='green', linestyle=':')
        
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir / f"{target}_distribution.png")
        plt.close()

def plot_results(data, df_reg_scores, df_cls_scores, feat_stats, params_summary, args):
    # Plot QQ
    # qq_plots_reg(df_reg_scores, args)
    # qq_plots_cls(df_cls_scores, args)

    plot_feature_target_scatter(data)

    plot_score_distributions_reg(df_reg_scores, args)
    plot_score_distributions_cls(df_cls_scores, args)

    corr_coefs = compute_corr_coefs(data)
    # Print mean correlation coefficient for regression targets
    reg_targets = [col for col in corr_coefs.columns if col in five_d_asc_list]
    mean_corr = corr_coefs[reg_targets].abs().mean().mean()
    print(f"\nMean absolute correlation coefficient between 5D-ASC scores: {mean_corr:.3f}")
    plot_feat_stats(feat_stats, corr_coefs)
    plot_5d_asc_distributions(data)