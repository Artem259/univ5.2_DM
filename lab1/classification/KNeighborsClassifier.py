import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import validate_data, check_is_fitted


class KNeighborsClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        X = np.array(X)

        self.classes_, y = np.unique(y, return_inverse=True)

        # TODO

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        # TODO
        return ...

    def kneighbors(self, X, n_neighbors):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        # TODO
        return ...
