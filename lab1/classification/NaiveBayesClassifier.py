import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import validate_data, check_is_fitted


class NaiveBayesClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags

    def fit(self, X, y):
        X, y = validate_data(self, X, y)

        if type_of_target(y) in ("continuous", "continuous-multioutput"):
            raise ValueError(f"Unknown label type: {type_of_target(y)}")
        self.classes_, y = np.unique(y, return_inverse=True)
        y_series = pd.Series(y)

        X_df = pd.DataFrame(X)

        # TODO

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        try:
            y_pred = ... # TODO
            return np.array(y_pred)
        except KeyError as e:
            raise KeyError(
                f"NaiveBayesClassifier encountered an unknown value '{e.args[0]}' in feature index {self.attr_i_}. "
                "Ensure that all input values were seen during training."
            )
