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

        self.distance_matrix_ = tools.calc_distance_matrix(X, X)
        num_samples = X.shape[0]
        labels = np.arange(num_samples)
        linkage_matrix = self._init_linkage_matrix(X)
        linkage_indices = [[i, [i]] for i in range(num_samples)]
        children = []
        distances = []

        for i in range(num_samples - 1):
            child, distance, linkage_matrix = self._merge_clusters_iter(X, labels, linkage_matrix, linkage_indices)
            children.append(child)
            distances.append(distance)

            if i == num_samples - self.n_clusters - 1:
                _, self.labels_ = np.unique(labels, return_inverse=True)

        self.children_ = np.array(children)
        self.distances_ = np.array(distances)
        return self

    def _init_linkage_matrix(self, X):
        if self.linkage == 'ward':
            linkage_matrix = ...  # TODO
        else:
            linkage_matrix = self.distance_matrix_.copy()

        np.fill_diagonal(linkage_matrix, np.inf)
        return linkage_matrix

    def _merge_clusters_iter(self, X, labels, linkage_matrix, linkage_indices):
        lm_min_index = tuple(sorted(np.unravel_index(np.argmin(linkage_matrix), linkage_matrix.shape)))
        index1, indices1 = linkage_indices[lm_min_index[0]]
        index2, indices2 = linkage_indices[lm_min_index[1]]
        child = [index1, index2]
        distance = linkage_matrix[lm_min_index]

        size = linkage_matrix.shape[0]
        if size == 1:
            return child, distance, None

        new_index = int(max(labels)) + 1
        new_indices = indices1 + indices2
        new_lm_array = np.array([
            self._calc_clusters_distance(X, new_indices, linkage_indices[i][1])
            if i != size else np.inf
            for i in range(size + 1)
            if i not in lm_min_index
        ])
        new_linkage_index = [new_index, new_indices]

        labels[np.isin(labels, [index1, index2])] = new_index
        linkage_matrix = np.delete(linkage_matrix, lm_min_index, axis=0)
        linkage_matrix = np.delete(linkage_matrix, lm_min_index, axis=1)
        linkage_matrix = np.pad(linkage_matrix, ((0, 1), (0, 1)))
        linkage_matrix[-1, :] = new_lm_array
        linkage_matrix[:, -1] = new_lm_array
        del linkage_indices[lm_min_index[1]]
        del linkage_indices[lm_min_index[0]]
        linkage_indices.append(new_linkage_index)

        return child, distance, linkage_matrix

    def _calc_clusters_distance(self, X, indices1, indices2):
        linkage_methods = {
            'single': self._single_clusters_distance,
            'complete': self._complete_clusters_distance,
            'average': self._average_clusters_distance,
            # 'ward': _ward_clusters_distance,  # TODO
        }
        return linkage_methods[self.linkage](X, indices1, indices2)

    def _single_clusters_distance(self, X, indices1, indices2):
        submatrix = self._get_distance_submatrix(indices1, indices2)
        return np.min(submatrix)

    def _complete_clusters_distance(self, X, indices1, indices2):
        submatrix = self._get_distance_submatrix(indices1, indices2)
        return np.max(submatrix)

    def _average_clusters_distance(self, X, indices1, indices2):
        submatrix = self._get_distance_submatrix(indices1, indices2)
        return np.mean(submatrix)

    # def _ward_clusters_distance(self, indices1, indices2, distance_matrix):  # TODO
    #     submatrix = _get_submatrix(distance_matrix, indices1, indices2)
    #     n1, n2 = len(indices1), len(indices2)
    #
    #     sum_squared_distances = np.sum(submatrix ** 2)
    #     return np.sqrt((n1 * n2) / (n1 + n2) * sum_squared_distances / (n1 * n2))

    def _get_distance_submatrix(self, indices1, indices2):
        return self.distance_matrix_[np.ix_(indices1, indices2)]

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
