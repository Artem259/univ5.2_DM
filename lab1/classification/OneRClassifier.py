import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import validate_data, check_is_fitted


class OneRClassifier(ClassifierMixin, BaseEstimator):
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

        num_samples = X.shape[0]
        best_error_rate = 1.01
        for feature_index, feature_values in enumerate(X.T):
            df = pd.DataFrame({'feat_v': feature_values, 'y': y})

            df_grouped = df.groupby('feat_v')['y']
            prediction_rules = df_grouped.apply(lambda x: x.value_counts().idxmax())
            accuracy = float(df_grouped.apply(lambda x: x.value_counts().max()).sum()) / num_samples
            error_rate = 1 - accuracy

            if error_rate < best_error_rate:
                best_error_rate = error_rate
                self.best_feature_index_ = feature_index
                self.prediction_rules_ = prediction_rules

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        try:
            predictions = [self.classes_[self.prediction_rules_[x[self.best_feature_index_]]] for x in X]
            return np.array(predictions)
        except KeyError as e:
            raise KeyError(
                f"OneRClassifier encountered an unknown value '{e.args[0]}' in feature index {self.best_feature_index_}. "
                "Ensure that all input values were seen during training."
            )
