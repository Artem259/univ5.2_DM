import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import validate_data

from . import tools


class DBSCAN(ClusterMixin, BaseEstimator):
    def __init__(self, eps=0.5, min_samples=5):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X, y=None):
        self.__validate_params()
        X = validate_data(self, X)
        X = np.array(X)

        # TODO

        return self

    def __validate_params(self):
        if not (isinstance(self.eps, float) or isinstance(self.eps, int)) or self.eps <= 0:
            raise ValueError(
                f"The 'eps' parameter must be a float in the range (0, inf). "
                f"Got '{self.eps}' instead."
            )
        if not isinstance(self.min_samples, int) or self.min_samples < 1:
            raise ValueError(
                f"The 'min_samples' parameter must be an int in the range [1, inf). "
                f"Got '{self.min_samples}' instead."
            )
