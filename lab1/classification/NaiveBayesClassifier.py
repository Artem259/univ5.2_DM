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

        self.class_probs_ = y_series.value_counts(normalize=True).to_dict()

        self.attr_probs_ = []
        self.attr_missing_probs_ = []
        for attr_i, attr_values in enumerate(X.T):
            attr_unique_num = len(np.unique(attr_values))
            y_attr_df = pd.DataFrame({'y': y, 'attr': attr_values})

            y_attr_grouped = y_attr_df.groupby('y')['attr']
            attr_probs = y_attr_grouped.apply(lambda x: x.value_counts())
            attr_probs = attr_probs.groupby(level=0).apply(lambda x: (x + 1) / (sum(x) + attr_unique_num))
            self.attr_probs_.append(attr_probs)

            attr_missing_prob = 1 / attr_unique_num
            self.attr_missing_probs_.append(attr_missing_prob)

        print(self.attr_probs_)
        print(self.attr_missing_probs_)
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
