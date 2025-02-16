import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import validate_data, check_is_fitted


class KNeighborsClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_neighbors=5, weights='uniform'):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        X = np.array(X)

        if self.weights not in ('distance', 'uniform'):
            raise ValueError(
                f"The 'weights' parameter of KNeighborsClassifier must be a str among ['distance', 'uniform']. "
                f"Got '{self.weights}' instead."
            )

        if type_of_target(y) in ("continuous", "continuous-multioutput"):
            raise ValueError(f"Unknown label type: {type_of_target(y)}")
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

        return self._kneighbors(X, n_neighbors)

    def _kneighbors(self, X, n_neighbors):
        # TODO
        return ...
