import numpy as np

from src.data_utils.splits import n_splits
from src.config import top_k_feats
from src.models.models import Model
from src.data_utils.fields import features
from src.utils.util_funcs import by_name, argbest
from src.utils.types import ranking, f_info
from src.config import model_params

def find_feats_for_fold(model: Model, train_sel_df, fold) -> f_info:
    train_mask = train_sel_df["sel_split"] != fold
    sel_mask = train_sel_df["sel_split"] == fold
    train_df = train_sel_df[train_mask]
    sel_df = train_sel_df[sel_mask]
    
    scores = np.empty(len(top_k_feats), dtype=float)
    params = model_params[model.__class__.__name__]
    default_params = {p_name: params[p_name]["default"] for p_name in params.keys()}
    model.set_important_params(final=False, **default_params)

    model.rank_features(train_df)
    for i, k in enumerate(top_k_feats):
        model.set_top_k_feats(k)
        model.fit(train_df)
        scores[i] = model.evaluate(sel_df)
    return scores, model.feat_ranking

def avg_feat_scores(feat_rankings: list[ranking]) -> ranking:
    feat_sums_dict = {f: 0 for f in features.keys()}
    for rnk in feat_rankings:
        for feat_name, score in rnk:
            feat_sums_dict[feat_name] += score
    feat_scores_avg = [(name, score / len(feat_rankings)) for name, score in feat_sums_dict.items()]
    return by_name(feat_scores_avg)

def vote_feats(suggestions: list[f_info]) -> f_info:
    ks_np = np.array([k for k, _ in suggestions])
    k_means = np.mean(ks_np, axis=0)
    feat_scores = avg_feat_scores([feat_scores for _, feat_scores in suggestions])
    return k_means, feat_scores

def select_features(model: Model, train_sel_df) -> f_info:
    suggestions: list[f_info] = [
        find_feats_for_fold(model, train_sel_df, fold)
        for fold in range(n_splits["inner"])
    ]
    k_means, best_feats = vote_feats(suggestions)
    best_k = top_k_feats[argbest(model)(k_means)]
    model.set_top_k_feats(best_k, best_feats)
    return k_means, best_feats
