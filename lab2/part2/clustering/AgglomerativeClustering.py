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

        num_samples = X.shape[0]
        labels = np.arange(num_samples)
        distance_matrix = tools.calc_distance_matrix(X, X)
        matrix_labels = labels.copy()
        children = []
        distances = []

        for i in range(num_samples):
            child, distance = self._merge_clusters_iter(X, labels, distance_matrix, matrix_labels)
            children.append(child)
            distances.append(distance)

            if i == num_samples - self.n_clusters:
                self.labels_ = np.array(labels)

        self.children_ = np.array(children)
        self.distances_ = np.array(distances)
        return self

    def _merge_clusters_iter(self, X, labels, distance_matrix, matrix_labels):
        ...  # TODO

    def _calc_clusters_distance(self, X, labels1, labels2):
        ...  # TODO

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
