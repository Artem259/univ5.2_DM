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

        records_num = X.shape[0]
        min_error_rate = 2.0
        for attr_i, attr_values in enumerate(X.T):
            df = pd.DataFrame({'attr_value': attr_values, 'y': y})

            df_grouped = df.groupby('attr_value')['y']
            rules = df_grouped.apply(lambda x: x.value_counts().idxmax())
            non_error_rate = float(df_grouped.apply(lambda x: x.value_counts().max()).sum()) / records_num
            error_rate = 1 - non_error_rate

            if error_rate < min_error_rate:
                min_error_rate = error_rate
                self.attr_i_ = attr_i
                self.rules_ = rules

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        try:
            y_pred = [self.classes_[self.rules_[x[self.attr_i_]]] for x in X]
            return np.array(y_pred)
        except KeyError as e:
            raise KeyError(
                f"OneRClassifier encountered an unknown value '{e.args[0]}' in feature index {self.attr_i_}. "
                "Ensure that all input values were seen during training."
            )
