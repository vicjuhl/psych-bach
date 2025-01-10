from abc import ABC, abstractmethod
import numpy as np

from src.data_utils.fields import sorted_features
from src.utils.types import ranking
from src.utils.util_funcs import by_score, by_name, check_lex_sorted, norm_rnk
from src.config import model_params

class Model(ABC):
    def __init__(self):
        self._model = None
        self._feats: list[str] = sorted_features
        self._feat_ranking: ranking = [(feat_name, None) for feat_name in self._feats] # None to avoid zero entries
        for param in model_params[self.__class__.__name__]:
            setattr(self, param, None)
    
    @property
    @abstractmethod
    def target(self) -> str:
        """The target variable name that this model predicts."""
        pass

    @property
    @abstractmethod
    def _score_func(self):
        """The scoring function used to evaluate model predictions."""
        pass

    @property
    def _important_params(self):
        """The parameters that can be tuned."""
        if not model_params[self.__class__.__name__]:
            return []
        else:
            return model_params[self.__class__.__name__].keys()
        
    @property
    def _current_params(self):
        """The current parameters of the model."""
        return {param: getattr(self, param) for param in self._important_params}

    @property
    def feat_ranking(self):
        return self._feat_ranking

    def get_important_params(self):
        """Get parameters important according to self.important_params."""
        return {param: getattr(self, param) for param in self._important_params}
    
    def set_important_params(self, final, **params): # final is intentionally unused due to non-kosher inheritance hack
        """Set attributes named as entries in self.important_params according to params."""
        for key, value in params.items():
            if hasattr(self, key) and key in self._important_params:
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")

    def set_top_k_feats(self, k, feat_ranking=None):
        """Set top k features to use for model fitting. If feat_ranking is not provided, use self._feat_ranking."""
        rnk = feat_ranking if feat_ranking is not None else self._feat_ranking
        check_lex_sorted(rnk)
        self._feats = [name for name, _ in by_name(by_score(rnk)[:k])]

    @abstractmethod
    def rank_features(self, train_df):
        """Rank features by importance, method depends on model."""
        pass

    def _set_default_params(self):
        """Set default parameters."""
        for param in model_params[self.__class__.__name__]:
            setattr(self, param, model_params[self.__class__.__name__][param]["default"])

    @abstractmethod
    def instantiate(self):
        """Instantiate the model."""
        pass

    @abstractmethod
    def _fit_implementation(self, train_df):
        """Concrete implementation of model fitting logic."""
        pass

    def _rank_by_corr_coefs(self, train_df) -> ranking:
        """Rank features by their correlation coefficients with the target variable.
        """
        # Calculate correlation coefficients between features and target
        X = train_df[sorted_features]
        y = train_df[self.target]
        
        # Calculate absolute correlation coefficient for each feature
        return norm_rnk([
            (f_name, abs(np.corrcoef(X[f_name], y.values.ravel())[0, 1]).item())
            for f_name in sorted_features
        ])

    def fit(self, train_df):
        """Template method that ensures common setup before specific fitting logic."""
        X = train_df[self._feats]
        y = np.ravel(train_df[self.target])
        
        check_lex_sorted(X.columns.tolist())
        
        # Instantiate and call the specific implementation
        self.instantiate()
        self._fit_implementation(X, y)

    @abstractmethod
    def predict(self, test_df):
        pass

    def evaluate(self, test_df, n_scrambles=None):
        """Evaluate model on test data."""
        X, y = test_df[self._feats], test_df[self.target]
        return float(self._score_func(self.predict(X), y))