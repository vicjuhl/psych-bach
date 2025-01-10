from src.config import (
    n_feats,
    meta_data,
    use_real_data,
    feature_set_suffixes
)

meta_fields = {
    "pid": int,
}

active_drug = {
    "active_drug": str,
}

five_d_asc_list = [
    "OSE",
    "AIA",
    "VUS",
    "AUD",
    "VIR"
]

target_cols = list(active_drug.keys()) + five_d_asc_list

five_d_asc = {
    name: float for name in five_d_asc_list
}

five_d_asc_limits = (0, 100)

labels = {
    **active_drug,
    **five_d_asc,
}

if use_real_data:
    features = {
        f_name: float
        for f_name in meta_data.keys()
        if f_name.startswith(feature_set_suffixes)
    }
else:
    features = {
        f"f{i}": float for i in range(1, n_feats + 1)
    }
sorted_features = sorted(features.keys())

fields = {
    **meta_fields,
    **labels,
    **features,
}

# Sorted lexicopraphically within each field type
sorted_fields = (
    sorted(meta_fields.keys()) +
    target_cols +
    sorted_features
)

long_model_names = {
    "random_forest_regressor": "Random Forest Regressor",
    "random_forest_classifier": "Random Forest Classifier",
    "svm": "SVM Classifier",
    "always_mean": "Mean Guesser",
    "lin_reg": "Linear Regression",
    "elastic_reg": "Elastic Net Regression",
    "logistic_regression": "Logistic Regression"
}