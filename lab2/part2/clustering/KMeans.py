import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import validate_data


class KMeans(ClusterMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        X = validate_data(self, X)
        X = np.array(X)

        # TODO

        self.labels_ = ...
        return self
