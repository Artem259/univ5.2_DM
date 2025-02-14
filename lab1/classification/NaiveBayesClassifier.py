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
        self.class_probs_ = np.log(y_series.value_counts(normalize=True)).sort_index()

        self.attr_unique_ = []
        self.attr_probs_ = []
        self.attr_missing_probs_ = []
        for attr_i, attr_values in enumerate(X.T):
            attr_unique = set(np.unique(attr_values).tolist())
            attr_unique_num = len(attr_unique)
            self.attr_unique_.append(attr_unique)

            df = pd.DataFrame({'y': y, 'attr': attr_values})
            df_grouped = df.groupby('y')['attr']
            df_grouped_counted = df_grouped.value_counts()
            attr_probs = df_grouped_counted.groupby(level=0).apply(
                lambda x: np.log((x + 1) / (sum(x) + attr_unique_num))
            ).reset_index(level=1, drop=True)
            self.attr_probs_.append(attr_probs)

            attr_missing_probs = df_grouped.apply(lambda x: np.log(1 / (len(x) + attr_unique_num)))
            self.attr_missing_probs_.append(attr_missing_probs)

        self.attrs_num_ = len(self.attr_unique_)

        return self

    def decision_function(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        return self.__decision_function(X)

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        D = self.__decision_function(X)
        return self.classes_[np.argmax(D, axis=1)]

    def __decision_function(self, X):
        attr_probs = pd.concat(self.attr_probs_, keys=range(self.attrs_num_))

        D = []
        for x in X:
            x_probs = attr_probs.groupby(level=[0, 1]).apply(
                lambda group: self.__aaa(group, x)
            )
            d = x_probs.groupby('y').sum() + self.class_probs_
            D.append(d)

        return np.array(D)

    def __aaa(self, group, x):
        attr_i = group.index.get_level_values(0).tolist()[0]
        group_y = group.index.get_level_values('y').tolist()[0]
        group_attr_values = group.index.get_level_values('attr').tolist()

        x_attr_v = x[attr_i]
        if x_attr_v in group_attr_values:
            return group.loc[attr_i, group_y, x_attr_v]
        if x_attr_v not in self.attr_unique_[attr_i]:
            raise KeyError(
                f"NaiveBayesClassifier encountered an unknown value '{x_attr_v}' in feature index {attr_i}. "
                "Ensure that all input values were seen during training."
            )
        return self.attr_missing_probs_[attr_i][group_y]
