from sklearn.metrics import confusion_matrix
import numpy as np

from src.utils.types import ranking

def upper_median(lst: list[int | float]) -> int | float:
    """Return the upper median of a descendingly sorted list of numbers."""
    return lst[len(lst) // 2]

def by_score(name_score: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Sort by score in descending order. First of tuple must be feature name, second must be score."""
    return sorted(tuple(name_score), key=lambda x: x[1], reverse=True)

def by_name(name_score: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Sort by name in lexicographic order. First of tuple must be feature name, second must be score."""
    return sorted(tuple(name_score), key=lambda x: x[0])

def norm_rnk(rnk: ranking) -> ranking:
    """Normalize ranking to sum to 1."""
    scores = [score for _, score in rnk]
    score_sum = sum(scores)
    return [(name, score / score_sum) for name, score in rnk]

def check_lex_sorted(lst):
    """Check if list is lexicographically sorted."""
    if lst == []:
        return
    has_iter_elms = hasattr(lst[0], '__iter__') and not isinstance(lst[0], str)
    lst_2 = [el[0] for el in lst] if has_iter_elms else lst

    if not all(lst_2[i] <= lst_2[i+1] for i in range(len(lst_2)-1)):
        raise ValueError(f"List is not lexicographically sorted: Got {lst}")
    
def get_confusion_matrix(model, test_df):
    y_pred = model.predict(test_df)
    y_true = test_df[model.target]
    return confusion_matrix(y_true, y_pred).tolist()

def scramble_labels(df, target_cols, rng):
    """Returns df with scrambled labels, not altering original df"""
    df_scr = df.copy()
    permuted_indices = rng.permutation(len(df_scr))
    df_scr[target_cols] = df_scr[target_cols].values[permuted_indices]
    return df_scr

def argbest(model):
    """Choose argmin or argmax depending on model type."""
    if model.__class__.__base__.__name__ == "Regressor":
        return np.argmin
    elif model.__class__.__base__.__name__ == "Classifier":
        return np.argmax
    else:
        raise ValueError(f"Model {model.__class__.__name__} is not a regressor or classifier")