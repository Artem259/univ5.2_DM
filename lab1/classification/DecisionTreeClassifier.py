import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import validate_data, check_is_fitted


class DecisionTreeClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        X = np.array(X)

        if type_of_target(y) in ("continuous", "continuous-multioutput"):
            raise ValueError(f"Unknown label type: {type_of_target(y)}")
        self.classes_, y = np.unique(y, return_inverse=True)

        self.num_features_ = X.shape[1]
        feat_indices = list(range(self.num_features_))
        df = pd.DataFrame(X, columns=feat_indices)
        df['y'] = y
        self.tree_ = self._id3_algorithm(df, set(feat_indices))

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        # TODO
        return ...  # self.classes_[np.argmax(decision_scores, axis=1)]

    def _id3_algorithm(self, df: pd.DataFrame, feat_indices: set[int]) -> "DecisionTreeNode":
        node = DecisionTreeNode()

        y_counts = df['y'].value_counts()
        most_frequent_y = y_counts.index[0]
        if len(y_counts) == 1 or not feat_indices:
            node.make_leaf(most_frequent_y)
            return node

        best_feat = max(feat_indices, key=lambda x: self._information_gain(df, x))
        node.make_splitter(best_feat)
        for best_feat_value, df_group in df.groupby(by=best_feat):
            if len(df_group) == 0:
                child_node = DecisionTreeNode()
                child_node.make_leaf(most_frequent_y)
            else:
                child_node = self._id3_algorithm(df_group, feat_indices - {best_feat})
            node.add_splitter_child(best_feat_value, child_node)

        return node

    def _information_gain(self, df: pd.DataFrame, feat_index: int) -> float:
        # TODO
        return 0.5


class DecisionTreeNode:
    def __init__(self):
        self.is_leaf = None

        # For splitter node
        self.feat_index = None
        self.children = None

        # For leaf node
        self.label = None

    def make_splitter(self, feat_index: int):
        self.is_leaf = False
        self.feat_index = feat_index
        self.children = {}
        self.label = None

    def make_leaf(self, label: int):
        self.is_leaf = True
        self.feat_index = None
        self.children = None
        self.label = label

    def add_splitter_child(self, feat_value, child_node: "DecisionTreeNode"):
        self.children[feat_value] = child_node
