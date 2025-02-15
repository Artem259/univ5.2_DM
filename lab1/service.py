import numpy as np
import sklearn

from classification.OneRClassifier import OneRClassifier
from classification.NaiveBayesClassifier import NaiveBayesClassifier
from classification.KNeighborsClassifier import KNeighborsClassifier


def OneRClassifier_info(clf: OneRClassifier):
    for rule in clf.prediction_rules_.items():
        print(f"x[{clf.best_feature_index_}] == {rule[0]}: y = {clf.classes_[rule[1]]}")


def sklearn_DecisionTreeClassifier_info(clf: sklearn.tree.DecisionTreeClassifier):
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    values = clf.tree_.value
    class_labels = clf.classes_

    # Root node (node 0)
    root_feature = feature[0]
    root_threshold = threshold[0]

    # Left child (x[root_feature] <= root_threshold)
    left_class_idx = np.argmax(values[children_left[0]])
    left_class_label = class_labels[left_class_idx]
    print(f"x[{root_feature}] <= {root_threshold:.1f}: y = {left_class_label}")

    # Right child (x[root_feature] > root_threshold)
    right_class_idx = np.argmax(values[children_right[0]])
    right_class_label = class_labels[right_class_idx]
    print(f"x[{root_feature}] > {root_threshold:.1f}: y = {right_class_label}")


def NaiveBayesClassifier_info(clf: NaiveBayesClassifier):
    info = {
        "class_log_probs_": clf.class_log_probs_,
        "feature_log_probs_": clf.feature_log_probs_,
        "feature_missing_log_probs_": clf.feature_missing_log_probs_,
    }
    for key, value in info.items():
        print(f"{key}:")
        print(*value, "\n")


def sklearn_CategoricalNB_info(clf: sklearn.naive_bayes.CategoricalNB):
    info = {
        "class_log_prior_": clf.class_log_prior_,
        "feature_log_prob_": clf.feature_log_prob_,
    }
    for key, value in info.items():
        print(f"{key}:")
        print(*value, "\n")


def KNeighborsClassifier_info(clf: KNeighborsClassifier, X_pred, n_neighbors):
    print("111")
    ... # TODO


def sklearn_KNeighborsClassifier_info(clf: sklearn.neighbors.KNeighborsClassifier, X_pred, n_neighbors):
    neigh_dist, neigh_ind = clf.kneighbors(X_pred, n_neighbors=n_neighbors)
    info = {
        "neighbors_dist": neigh_dist,
        "neighbors_indices": neigh_ind,
    }
    for key, value in info.items():
        print(f"{key}:")
        print(*value, "\n")
