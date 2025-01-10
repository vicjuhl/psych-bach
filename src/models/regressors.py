import numpy as np
from abc import ABC
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from src.models.models import Model
from src.data_utils.fields import sorted_features, five_d_asc_limits
from src.utils.util_funcs import by_name, norm_rnk
from src.config import n_trees_final

class Regressor(Model, ABC):
    def __init__(self):
        super().__init__()
        self._target = None

    @property
    def target(self) -> str:
        return self._target
    
    @target.setter
    def target(self, target: str):
        self._target = target

    @property
    def _score_func(self):
        return mean_squared_error
    
    def predict(self, test_df):
        X = test_df[self._feats]
        y_pred = self._model.predict(X)
        y_pred = np.clip(y_pred, *five_d_asc_limits)
        return y_pred

class _MeanGuesser:
    def __init__(self, target):
        self._mean = None

    def fit(self, y):
        self._mean = y.mean()

    def predict(self, df):
        return np.ones(len(df)) * self._mean

class MeanGuesser(Regressor):    
    def __init__(self):
        super().__init__()
        self._feats = []
        self._feat_ranking = []

    def instantiate(self):
        self._model = _MeanGuesser(self._target)

    def _fit_implementation(self, _, y):
        self._model.fit(y)

    def rank_features(self, _):
        """Assign equal rank -1 to all features."""
        pass
    
class LinReg(Regressor):
    def rank_features(self, train_df): # TODO proper sharing with LogReg?
        by_corr = self._rank_by_corr_coefs(train_df)
        # Set n-1 features according to ranking by mutual information scores between features and target
        self.set_top_k_feats(len(train_df) - 1, by_corr)
        names = [f_name for f_name in self._feats]

        self.fit(train_df)  # assumes that the features are sorted lexicographically in df
        abs_w = [abs(c.item()) for c in self._model.coef_]  # by name
        self._feat_ranking = norm_rnk(by_name(zip(names, abs_w)))

    def instantiate(self):
        self._model = LinearRegression(fit_intercept=True, copy_X=True)

    def _fit_implementation(self, X, y):
        self._model.fit(X, y)

class ElasticReg(Regressor):
    def __init__(self, seeds):
        super().__init__()
        self._seeds = seeds

    def rank_features(self, train_df):
        # Get correlation-based ranking
        by_corr = self._rank_by_corr_coefs(train_df)
        self.set_top_k_feats(len(train_df) - 1, by_corr)
        names = [f_name for f_name in self._feats]

        self._set_default_params()
        self.fit(train_df)  # assumes that the features are sorted lexicographically in df
        
        # Get absolute weights and normalize them
        abs_w = [abs(c.item()) for c in self._model.coef_]  # by name
        self._feat_ranking = norm_rnk(by_name(zip(names, abs_w)))

    def instantiate(self):
        self._model = ElasticNet(
            **self._current_params,
            max_iter=2000,
            fit_intercept=True,
            copy_X=True,
            random_state=next(self._seeds)
        )

    def _fit_implementation(self, X, y):
        self._model.fit(X, y)

class RFReg(Regressor):
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
        self._model = RandomForestRegressor(
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

def instantiate_regressors(rand_iterator):
    return {
        "always_mean": MeanGuesser(),
        "lin_reg": LinReg(),
        "elastic_reg": ElasticReg(rand_iterator),
        "random_forest_regressor": RFReg(rand_iterator)
    }