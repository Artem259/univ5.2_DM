import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class OneRClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self):
        ...

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        X, y = np.array(X), np.array(y)
        self.classes_ = unique_labels(y)

        # TODO

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        # TODO

        return ...
