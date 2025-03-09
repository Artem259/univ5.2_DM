import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import validate_data

from . import tools


class AgglomerativeClustering(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters=2, linkage='ward'):
        super().__init__()
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X, y=None):
        self.__validate_params()
        X = validate_data(self, X)
        X = np.array(X)

        # TODO

        return self

    def __validate_params(self):
        if not isinstance(self.n_clusters, int) or self.n_clusters < 1:
            raise ValueError(
                f"The 'n_clusters' parameter must be an int in the range [1, inf). "
                f"Got '{self.n_clusters}' instead."
            )
        if self.linkage not in ('single', 'complete', 'average', 'ward'):
            raise ValueError(
                f"The 'linkage' parameter must be a str among "
                f"['single', 'complete', 'average', 'ward']. Got '{self.linkage}' instead."
            )
