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
        X = np.array(X)

        if type_of_target(y) in ("continuous", "continuous-multioutput"):
            raise ValueError(f"Unknown label type: {type_of_target(y)}")
        self.classes_, y = np.unique(y, return_inverse=True)

        y_series = pd.Series(y)
        self.class_probs_ = np.log(y_series.value_counts(normalize=True)).sort_index().tolist()

        self.attr_probs_ = []
        self.attr_missing_probs_ = []
        for attr_i, attr_values in enumerate(X.T):
            attr_unique_num = len(np.unique(attr_values))
            df = pd.DataFrame({'y': y, 'attr': attr_values})

            df_grouped = df.groupby('y')['attr']
            df_grouped_counted = df_grouped.value_counts()
            attr_probs = df_grouped_counted.groupby(level=0).apply(
                lambda x: np.log((x + 1) / (sum(x) + attr_unique_num))
            )
            self.attr_probs_.append(attr_probs)

            attr_missing_probs = df_grouped.apply(lambda x: np.log(1 / (len(x) + attr_unique_num)))
            self.attr_missing_probs_.append(attr_missing_probs)

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
