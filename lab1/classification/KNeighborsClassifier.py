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
        self.__validate_params()
        X, y = validate_data(self, X, y)
        X = np.array(X)

        if type_of_target(y) in ("continuous", "continuous-multioutput"):
            raise ValueError(f"Unknown label type: {type_of_target(y)}")
        self.classes_, y = np.unique(y, return_inverse=True)

        self.fitted_X_ = X
        self.fitted_y_ = y

        return self

    def predict(self, X):
        self.__validate_params()
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        decision_scores = self._decision_function(X)
        print(decision_scores)  # TODO remove
        return self.classes_[np.argmax(decision_scores, axis=1)]

    def kneighbors(self, X):
        self.__validate_params()
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        return self._kneighbors(X)

    def _decision_function(self, X):
        self.__validate_params()

        decision_scores = []
        for x in X:
            if self.weights == 'distance':
                x_decision_scores = ...  # TODO
            elif self.weights == 'uniform':
                x_neigh_indices = self._find_kneighbors_indices(x, self.n_neighbors)
                x_neigh_labels = self.fitted_y_[x_neigh_indices]
                x_decision_scores = np.bincount(x_neigh_labels, minlength=len(self.classes_))
            else:
                raise RuntimeError("This code should never be reached.")
            decision_scores.append(x_decision_scores)

        return np.array(decision_scores)

    def _kneighbors(self, X):
        neigh_distances = []
        neigh_indices = []
        for x in X:
            x_neigh_indices = self._find_kneighbors_indices(x, self.n_neighbors)
            x_neigh_distances, _ = self._calc_distances(x, X_to=self.fitted_X_[x_neigh_indices])
            neigh_distances.append(x_neigh_distances)
            neigh_indices.append(x_neigh_indices)

        return np.array(neigh_distances), np.array(neigh_indices)

    def _find_kneighbors_indices(self, x, n_neighbors):
        indices = list(range(self.fitted_X_.shape[0]))
        _, distances_squared = self._calc_distances(x)
        neigh_indices = sorted(indices, key=lambda i: distances_squared[i])[:n_neighbors]
        return np.array(neigh_indices)

    def _calc_distances(self, x_from, X_to=None):
        if X_to is None:
            X_to = self.fitted_X_
        distances_squared = np.sum((X_to - x_from) ** 2, axis=1)
        distances = np.sqrt(distances_squared)
        return distances, distances_squared

    def __validate_params(self):
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
