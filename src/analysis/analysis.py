from src.utils.prints import (
    print_dict, 
    print_comparison_matrix, 
    print_reg_scores_stats,
    print_cls_scores_stats,
    print_k_stats
)
from src.data_utils.imports_exports import (
    load_results,
    load_data
)
from src.data_utils.preparation import whiten_all
from src.analysis.summaries import (
    summarize_params, 
    aggregate_scores_reg, 
    aggregate_scores_cls, 
    aggregate_feature_importance,
    summarize_confusion_matrices,
)
from src.analysis.statistics import (
    compute_cls_pairwise_pvalues,
    compute_reg_pairwise_pvalues,
    compute_scrambled_p_values_reg,
    compute_scrambled_p_values_cls,
    compute_feature_stats,
    compute_regressor_p_values,
)
from src.analysis.plots import plot_results
from src.config import data_path, results_path

import pandas as pd

def analyze_results(args):
    results = load_results(args['results_pattern'])
    data = whiten_all(load_data(data_path / "data.csv"))

    # Performance scores
    df_reg_scores = aggregate_scores_reg(results["regressors"])
    df_cls_scores = aggregate_scores_cls(results["classifiers"])

    print_reg_scores_stats(df_reg_scores[df_reg_scores['scrambled'] == False])
    print_cls_scores_stats(df_cls_scores[df_cls_scores['scrambled'] == False])

    # Feature importances
    df_feats = aggregate_feature_importance(results)
    print_k_stats(df_feats)
    print("\nFeature importances:\n")
    feat_path = results_path / 'latest_feature_stats.csv'
    if args["fetch_feat_imps"]:
        print("IMPORTING FEATURE STATISTICS FROM PREVIOUS RUN")
        feat_stats = pd.read_csv(feat_path)
    else:
        feat_stats = compute_feature_stats(df_feats)
        feat_stats.to_csv(feat_path, index=False)
    # print_feature_stats(feat_stats, 15)

    # Parameters
    params_summary = summarize_params(results)
    print("\nParameters:\n")
    print_dict(params_summary)

    # Confusion matrices
    conf_matrices = summarize_confusion_matrices(results)
    print("\nConfusion matrices:\n")
    print_dict(conf_matrices)

    # Pairwise p-values for classifiers
    cls_pvalues, cls_name_mapping = compute_cls_pairwise_pvalues(df_cls_scores)
    print_comparison_matrix(cls_pvalues, cls_name_mapping, "Classifier performance p-values")

    # Pairwise p-values for regressors
    reg_pvalues, reg_name_mapping = compute_reg_pairwise_pvalues(df_reg_scores)
    for target, matrix in reg_pvalues.items():  
        print_comparison_matrix(matrix, reg_name_mapping, "Regressor performance p-values", target)

    print("\np-values for models performing significantly better than their scrambled counterparts")
    print("regressors:")
    reg_scramble_p_values = compute_scrambled_p_values_reg(df_reg_scores)
    print(reg_scramble_p_values)

    cls_scramble_p_values = compute_scrambled_p_values_cls(df_cls_scores)
    print("\nclassifiers")
    print(cls_scramble_p_values)

    print("\nAssessing wheather the mean guesser outperforms the actual models")
    print(compute_regressor_p_values(df_reg_scores, greater=True))
    print("\nAssessing wheather the actual models outperforms the mean guesser")
    print(compute_regressor_p_values(df_reg_scores, greater=False))

    # Plots
    if args['plot']:
        plot_results(data, df_reg_scores, df_cls_scores, feat_stats, params_summary, args)