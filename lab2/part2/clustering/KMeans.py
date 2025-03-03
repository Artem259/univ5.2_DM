import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import validate_data

from . import tools


class KMeans(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters, init, max_iter=300, e=1e-4):
        super().__init__()
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.e = e

    def fit(self, X, y=None):
        self.__validate_params()
        X = validate_data(self, X)
        X = np.array(X)

        self.cluster_centers_ = np.array(self.init)
        self.n_iter_ = 0
        while self.n_iter_ < self.max_iter:
            prev_cluster_centers = self.cluster_centers_
            # TODO
            max_centers_dist_diff = tools.calc_max_zip_distance(
                self.cluster_centers_,
                prev_cluster_centers,
                tools.euclidean_distance
            )
            if max_centers_dist_diff < self.e:
                break

        self.labels_ = ...  # TODO
        return self

    def __validate_params(self):
        if not isinstance(self.n_clusters, int) or self.n_clusters < 1:
            raise ValueError(
                f"The 'n_clusters' parameter of KMeans must be an int in the range [1, inf). "
                f"Got '{self.n_clusters}' instead."
            )
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError(
                f"The 'max_iter' parameter of KMeans must be an int in the range [1, inf). "
                f"Got '{self.max_iter}' instead."
            )
        if not (isinstance(self.e, float) or isinstance(self.e, int)) or self.e < 0:
            raise ValueError(
                f"The 'e' parameter of KMeans must be a float in the range [0, inf). "
                f"Got '{self.e}' instead."
            )

        init_shape = np.array(self.init).shape
        if init_shape[0] != self.n_clusters:
            raise ValueError(
                f"The shape of the initial centers {init_shape} "
                f"does not match the number of clusters {self.n_clusters}."
            )
