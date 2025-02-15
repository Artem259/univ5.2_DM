import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
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

        self.classes_, y = np.unique(y, return_inverse=True)
        y_series = pd.Series(y)
        self.class_log_probs_ = np.log(y_series.value_counts(normalize=True)).sort_index()

        self.feature_unique_values_ = []
        self.feature_log_probs_ = []
        self.feature_missing_log_probs_ = []
        for feature_index, feature_values in enumerate(X.T):
            unique_values = set(np.unique(feature_values).tolist())
            unique_num = len(unique_values)
            self.feature_unique_values_.append(unique_values)

            df = pd.DataFrame({'y': y, 'feat_v': feature_values})
            df_grouped = df.groupby('y')['feat_v']
            grouped_counts = df_grouped.value_counts()
            log_probs = grouped_counts.groupby(level=0).apply(
                lambda x: np.log((x + 1) / (sum(x) + unique_num))
            ).reset_index(level=1, drop=True)
            self.feature_log_probs_.append(log_probs)

            missing_log_probs = df_grouped.apply(
                lambda x: np.log(1 / (len(x) + unique_num))
            )
            self.feature_missing_log_probs_.append(missing_log_probs)

        self.num_features_ = len(self.feature_unique_values_)
        return self

    def decision_function(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        return self._decision_function(X)

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        decision_scores = self._decision_function(X)
        return self.classes_[np.argmax(decision_scores, axis=1)]

    def _decision_function(self, X):
        feature_probs_concat = pd.concat(self.feature_log_probs_, keys=range(self.num_features_))

        decision_scores = []
        for x in X:
            x_feature_log_probs = feature_probs_concat.groupby(level=[0, 'y']).apply(
                lambda group: self._insert_missing_probs(group, x)
            )
            x_decision_scores = x_feature_log_probs.groupby('y').sum() + self.class_log_probs_
            decision_scores.append(x_decision_scores)

        return np.array(decision_scores)

    def _insert_missing_probs(self, group, x):
        feature_index = group.index.get_level_values(0).tolist()[0]
        y_value = group.index.get_level_values('y').tolist()[0]
        known_values = group.index.get_level_values('feat_v').tolist()

        feature_value = x[feature_index]
        if feature_value in known_values:
            return group.loc[feature_index, y_value, feature_value]
        if feature_value not in self.feature_unique_values_[feature_index]:
            raise KeyError(
                f"NaiveBayesClassifier encountered an unknown value '{feature_value}' in feature index {feature_index}. "
                "Ensure that all input values were seen during training."
            )
        return self.feature_missing_log_probs_[feature_index][y_value]
