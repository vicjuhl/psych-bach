from src.config import n_scrambles
from src.data_utils.fields import five_d_asc_list

def get_empty_results(n_experiments, regressor_names, classifier_names):
    return {
        "classifiers": [
            {
                name: {
                    "scores": {"true": [], "scrambled": [[] for _ in range(n_scrambles)]},
                    "features": [],
                    "params": [],
                    "confusion_matrices": []
                } for name in classifier_names
            } for _ in range(n_experiments)
        ], "regressors": [
            {
                target: {
                    name: {
                        "scores": {"true": [], "scrambled": [[] for _ in range(n_scrambles)]},
                        "features": [],
                        "params": []
                    } for name in regressor_names
                } for target in five_d_asc_list
            } for _ in range(n_experiments)
        ]
    }