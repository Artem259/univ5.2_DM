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

        if not isinstance(self.n_neighbors, int) or self.n_neighbors < 1:
            raise ValueError(
                f"The 'n_neighbors' parameter of KNeighborsClassifier must be an int in the range [1, inf). "
                f"Got '{self.n_neighbors}' instead."
            )
        if self.weights not in ('distance', 'uniform'):
            raise ValueError(
                f"The 'weights' parameter of KNeighborsClassifier must be a str among ['distance', 'uniform']. "
                f"Got '{self.weights}' instead."
            )

        if type_of_target(y) in ("continuous", "continuous-multioutput"):
            raise ValueError(f"Unknown label type: {type_of_target(y)}")
        self.classes_, y = np.unique(y, return_inverse=True)

        self.fitted_X_ = X

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        # TODO
        return ...

    def kneighbors(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        return self._kneighbors(X, self.n_neighbors)

    def _kneighbors(self, X, n_neighbors):
        indices = list(range(self.fitted_X_.shape[0]))

        neigh_distances = []
        neigh_indices = []
        for x in X:
            x_distances = np.sqrt(np.sum((self.fitted_X_ - x) ** 2, axis=1))
            x_neigh_indices = sorted(indices, key=lambda i: x_distances[i])[:n_neighbors]
            x_neigh_distances = x_distances[x_neigh_indices]
            neigh_distances.append(x_neigh_distances)
            neigh_indices.append(x_neigh_indices)

        return np.array(neigh_distances), np.array(neigh_indices)
