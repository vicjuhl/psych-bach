import numpy as np
from typing import Any

from concurrent.futures import ProcessPoolExecutor
from src.data_utils.splits import n_splits, get_splits
from src.data_utils.preparation import get_data, active_only, whiten
from src.param_search import tune_model
from src.config import n_runs, n_cores, n_scrambles, seed_iterators, model_params
from src.data_utils.imports_exports import dump_results
from src.utils.util_funcs import get_confusion_matrix, scramble_labels
from src.data_utils.fields import five_d_asc_list, active_drug
from src.data_utils.types import get_empty_results
from src.models.models import Model
from src.models.regressors import instantiate_regressors
from src.models.classifiers import instantiate_classifiers

def train(model: Model, train_val_df) -> tuple[list, dict]:
    """Train model using outer CV for hyperparameter tuning (when relevant) and retrain with best hyperparams on full training/validation set."""
    (k, feat_ranking), params = tune_model(model, train_val_df)
    model.set_important_params(final=True, **params)
    model.set_top_k_feats(k, feat_ranking)
    model.fit(train_val_df)
    return (k, feat_ranking), params

def get_fold_results_reg(local_results, model: Model, train_data, test_data, scrambled, scr_idx):
    if scrambled:
        if model.__class__.__name__ == "MeanGuesser":
            return
        (_, _), _ = train(model, train_data)
        local_results["scores"]["scrambled"][scr_idx].append(model.evaluate(test_data))

    else:
        # Train
        (k, feat_ranking), p = train(model, train_data)
        # Score and gather feature/parameter choices
        local_results["scores"]["true"].append(model.evaluate(test_data))
        local_results["features"].append({
            "k": k,
            "feat_ranking": feat_ranking
        })
        local_results["params"].append(p)

def get_fold_results_cls(local_results, model: Model, train_data, test_data, scrambled, scr_idx):
    get_fold_results_reg(local_results, model, train_data, test_data, scrambled, scr_idx)
    if not scrambled:
        local_results["confusion_matrices"].append(get_confusion_matrix(model, test_data))

def cross_test(df, models: dict[str, Model], local_results, fold_results_fn, scrambled, scr_idx):
    for test_fold in range(n_splits["outer"]):
        train_mask = df["test_split"] != test_fold
        test_mask = df["test_split"] == test_fold
        train_data, test_data = whiten(df.copy(), train_mask, test_mask)
        # print(f"  Test fold {test_fold}")
        for name, model in models.items():
            # print(f"\n--------- {name.upper()} ---------")
            fold_results_fn(local_results[name], model, train_data, test_data, scrambled, scr_idx)

def run_one_experiment(i, df_folds, regressors, classifiers, results, scrambled, scr_idx=None, rng=None):
    # REGRESSORS
    for target in five_d_asc_list:
        if scrambled:
            # Scramble only 5D-ASC scores for regression
            df_to_use = scramble_labels(active_only(df_folds), five_d_asc_list, rng)
        else:
            df_to_use = active_only(df_folds)
            
        for reg in regressors.values():
            reg.target = target
        cross_test(df_to_use, regressors, results["regressors"][i][target], get_fold_results_reg, scrambled, scr_idx)

    # CLASSIFIERS
    if scrambled:
        # Scramble only active_drug for classification
        df_to_use = scramble_labels(df_folds, list(active_drug.keys()), rng)
    else:
        df_to_use = df_folds
        
    cross_test(df_to_use, classifiers, results["classifiers"][i], get_fold_results_cls, scrambled, scr_idx)

def run_n_experiments(df, n_experiments: int, rand_iterator, core_num) -> dict[str, dict[str, Any]]:
    """Run the pipeline n times and return the mean and std of the results."""
    regressors = instantiate_regressors(rand_iterator)
    classifiers = instantiate_classifiers(rand_iterator)
    results = get_empty_results(n_experiments, regressors.keys(), classifiers.keys())

    for i in range(n_experiments):
        print(f"\n----- EXPERIMENT {i + 1} for core {core_num + 1} -----")
        df_folds = get_splits(df, rand_iterator)
        # Unscrambled
        run_one_experiment(i, df_folds, regressors, classifiers, results, scrambled=False)
        
        # Scrambled
        for scr_idx in range(n_scrambles):
            rng = np.random.RandomState(next(rand_iterator))
            run_one_experiment(i, df_folds, regressors, classifiers, results, True, scr_idx, rng)
    
    return results

def execute_in_parallel(df, n_runs, rand_iterators):
    # Calculate how many experiments each core should run
    runs_per_core = n_runs // n_cores  # Since len(rand_iterators) == n_cores
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = [
            executor.submit(run_n_experiments, df, runs_per_core, rand_iterator, core_num)
            for core_num, rand_iterator in enumerate(rand_iterators)
        ]
        results = [future.result() for future in futures]
    # Combine results from all cores into one dictionary
    combined_results = {
        "classifiers": [],
        "regressors": []
    }
    # Merge results
    for result in results:
        combined_results["classifiers"].extend(result["classifiers"])
        combined_results["regressors"].extend(result["regressors"])
    return combined_results

def run_pipeline(args):
    """Run the pipeline."""
    df = get_data(args)
    results = execute_in_parallel(df, n_runs, seed_iterators)
    dump_results(results, args)

