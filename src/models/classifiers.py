from abc import ABC
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.data_utils.fields import sorted_features, active_drug
from src.models.models import Model
from src.utils.util_funcs import by_name, norm_rnk
from src.config import n_trees_final

class Classifier(Model, ABC):
    @property
    def target(self):
        return active_drug.keys()
    
    @property
    def _score_func(self):
        return f1_score
    
    def predict(self, test_df):
        X = test_df[self._feats]
        return self._model.predict(X)

class LogisticClassifier(Classifier):
    def __init__(self, seeds):
        super().__init__()
        self._seeds = seeds

    def instantiate(self):
        self._model = LogisticRegression(
            **self._current_params,
            fit_intercept=True,
            max_iter=2000,
            random_state=next(self._seeds)
        )

    def rank_features(self, train_df):
        # Rank by correlation with target and set top k=n-1 features as initial filtering
        by_corr = self._rank_by_corr_coefs(train_df)
        self.set_top_k_feats(len(train_df) - 1, by_corr)
        names = [f_name for f_name in self._feats]

        # Fit model
        self.fit(train_df) # assumes that the features are sorted lexicographically in df
        abs_w = [abs(c.item()) for c in self._model.coef_[0]] # by name

        # Rank by absolute weights
        self._feat_ranking = norm_rnk(by_name(zip(names, abs_w)))
        
    def _fit_implementation(self, X, y):
        self._model.fit(X, y)

class RFClassifier(Classifier):
    def __init__(self, seeds):
        super().__init__()
        self._seeds = seeds

    def set_important_params(self, final: bool, **params):
        """Set attributes named as entries in self.important_params according to params."""
        for key, value in params.items():
            if hasattr(self, key) and key in self._important_params:
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        if final:
            self.n_estimators = n_trees_final

    def instantiate(self):
        self._model = RandomForestClassifier(
            **self._current_params,
            random_state=next(self._seeds)
        )

    def rank_features(self, train_df):
        """Rank features by average impurity decrease across all trees."""
        self._set_default_params()
        self._feats = sorted_features
        self.fit(train_df)
        scores = [score.item() for score in self._model.feature_importances_]
        self._feat_ranking = norm_rnk(by_name(zip(sorted_features, scores)))

    def _fit_implementation(self, X, y):
        self._model.fit(X, y)

class SVMClassifier(Classifier):
    def __init__(self, seeds):
        super().__init__()
        self._seeds = seeds

    def instantiate(self):
        self._model = SVC(
            **self._current_params,
            kernel="linear",
            random_state=next(self._seeds)
        )
    
    def rank_features(self, train_df):
        # First rank by correlation coefficients
        by_corr = self._rank_by_corr_coefs(train_df)
        self.set_top_k_feats(len(train_df) - 1, by_corr)
        names = [f_name for f_name in self._feats]
        
        # Fit the model
        self._set_default_params()
        self.fit(train_df)
        abs_w = [abs(c.item()) for c in self._model.coef_[0]] # by name
        
        self._feat_ranking = norm_rnk(by_name(zip(names, abs_w)))
    
    def _fit_implementation(self, X, y):
        self._model.fit(X, y)

def instantiate_classifiers(rand_iterator):
    return {
        "logistic_regression": LogisticClassifier(rand_iterator),
        "random_forest_classifier": RFClassifier(rand_iterator),
        "svm": SVMClassifier(rand_iterator)
    }